use anyhow::{anyhow, Result};
use reqwest::Client;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::backends::BackendDescriptor;
use crate::budget::{
    format_bytes, headroom_bytes, measure_prompt_budget, usage_ratio, PromptBudget,
};
use crate::config::{AgentConfig, ContextConfig};
use crate::openai::{stream_chat, ChatMessage, ChatRequest, StreamOptions, ToolDef};

const SUMMARIZE_BUDGET_FRACTION: f64 = 0.5;
const TIER2_TOOL_MAX_CHARS: usize = 1500;
const SUMMARY_MARKER: &str = "\n\nConversation summary:\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactMethod {
    None,
    LlmSummary,
    DeterministicTrim,
}

#[derive(Debug, Clone)]
pub struct CompactResult {
    pub compacted: bool,
    pub before_messages: usize,
    pub after_messages: usize,
    pub before_ratio: f64,
    pub after_ratio: f64,
    pub method: CompactMethod,
    pub conversation_summary: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ContextGuardConfig {
    pub effective_limit_bytes: usize,
    pub compact_threshold: f64,
    pub auto_compact: bool,
    pub keep_messages: usize,
    pub model_context_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ContextGuardParams {
    pub effective_limit_bytes: usize,
    pub compact_threshold: f64,
    pub auto_compact: bool,
    pub keep_messages: usize,
    pub summarize_budget_bytes: usize,
    pub conversation_summary: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CompactNotice {
    pub line: String,
    pub conversation_summary: Option<String>,
    pub transcript_rewritten: bool,
}

#[derive(Debug)]
pub struct CompactSessionContext<'a> {
    pub messages: &'a mut Vec<ChatMessage>,
    pub system_prompt: &'a str,
    pub tool_defs: &'a [ToolDef],
    pub config: &'a AgentConfig,
    pub model: &'a str,
    pub is_local: bool,
    pub http: &'a Client,
    pub backend: &'a BackendDescriptor,
    pub conversation_summary: Option<&'a str>,
}

struct CompactRequest<'a> {
    messages: &'a mut Vec<ChatMessage>,
    system_prompt: &'a str,
    tool_defs: &'a [ToolDef],
    keep_messages: usize,
    limit_bytes: usize,
    summarize_budget_bytes: usize,
    http: &'a Client,
    backend: &'a BackendDescriptor,
    model: &'a str,
    conversation_summary: Option<&'a str>,
}

pub fn default_model_context_tokens(model: &str, is_local: bool) -> usize {
    let lower = model.to_ascii_lowercase();
    if !is_local {
        return 128_000;
    }
    if lower.contains("qwen2.5-coder") || lower.contains("qwen2.5_coder") {
        return 32_768;
    }
    if lower.contains("32k") {
        return 32_768;
    }
    if lower.contains("16k") {
        return 16_384;
    }
    if lower.contains("8k") {
        return 8192;
    }
    8192
}

pub fn resolve_model_context_tokens(config: &ContextConfig, model: &str, is_local: bool) -> usize {
    config
        .model_context_tokens
        .unwrap_or_else(|| default_model_context_tokens(model, is_local))
}

pub fn resolve_auto_compact(config: &ContextConfig, is_local: bool) -> bool {
    config.auto_compact.unwrap_or(is_local)
}

pub fn effective_limit_bytes(config: &ContextConfig, model: &str, is_local: bool) -> usize {
    let model_tokens = resolve_model_context_tokens(config, model, is_local);
    let fill_ratio = 1.0 - config.reserve_ratio.clamp(0.05, 0.5);
    let from_model = (model_tokens as f64 * 4.0 * fill_ratio) as usize;
    match config.max_bytes {
        Some(max_bytes) => from_model.min(max_bytes),
        None => from_model,
    }
}

pub fn guard_config_from(config: &AgentConfig, model: &str, is_local: bool) -> ContextGuardConfig {
    ContextGuardConfig {
        effective_limit_bytes: effective_limit_bytes(&config.context, model, is_local),
        compact_threshold: config.context.compact_threshold,
        auto_compact: resolve_auto_compact(&config.context, is_local),
        keep_messages: config.context.max_messages.unwrap_or(12).clamp(4, 80),
        model_context_tokens: resolve_model_context_tokens(&config.context, model, is_local),
    }
}

pub fn guard_params_from(
    config: &AgentConfig,
    model: &str,
    is_local: bool,
    conversation_summary: Option<String>,
) -> ContextGuardParams {
    let guard = guard_config_from(config, model, is_local);
    ContextGuardParams {
        effective_limit_bytes: guard.effective_limit_bytes,
        compact_threshold: guard.compact_threshold,
        auto_compact: guard.auto_compact,
        keep_messages: guard.keep_messages,
        summarize_budget_bytes: (guard.effective_limit_bytes as f64 * SUMMARIZE_BUDGET_FRACTION)
            as usize,
        conversation_summary,
    }
}

pub fn should_compact(budget: &PromptBudget, limit_bytes: usize, threshold: f64) -> bool {
    if limit_bytes == 0 {
        return false;
    }
    usage_ratio(budget, limit_bytes) >= threshold.clamp(0.5, 0.99)
}

pub fn format_usage_line(budget: &PromptBudget, limit_bytes: usize) -> String {
    format!(
        "{} ({:.0}% of {})",
        format_bytes(budget.effective_total_bytes),
        usage_ratio(budget, limit_bytes) * 100.0,
        format_bytes(limit_bytes)
    )
}

pub fn merge_system_prompt(base: &str, summary: Option<&str>) -> String {
    let base = strip_conversation_summary(base);
    match summary.map(str::trim).filter(|s| !s.is_empty()) {
        Some(summary) => format!("{base}{SUMMARY_MARKER}{summary}"),
        None => base,
    }
}

pub fn strip_conversation_summary(content: &str) -> String {
    content
        .split_once(SUMMARY_MARKER)
        .map(|(base, _)| base)
        .unwrap_or(content)
        .trim_end()
        .to_string()
}

pub fn extract_conversation_summary(content: &str) -> Option<String> {
    content
        .split_once(SUMMARY_MARKER)
        .map(|(_, summary)| summary.trim().to_string())
        .filter(|summary| !summary.is_empty())
}

pub fn validate_transcript(messages: &[ChatMessage]) -> Result<()> {
    let mut pending_tools = HashSet::new();
    for message in messages {
        match message {
            ChatMessage::System { .. } => {}
            ChatMessage::User { .. } => {
                if !pending_tools.is_empty() {
                    return Err(anyhow!(
                        "unfinished tool calls before user message: {pending_tools:?}"
                    ));
                }
            }
            ChatMessage::Assistant { tool_calls, .. } => {
                if !pending_tools.is_empty() {
                    return Err(anyhow!(
                        "unfinished tool calls before assistant message: {pending_tools:?}"
                    ));
                }
                for tool_call in tool_calls {
                    if !tool_call.id.is_empty() {
                        pending_tools.insert(tool_call.id.clone());
                    }
                }
            }
            ChatMessage::Tool { tool_call_id, .. } => {
                if !pending_tools.remove(tool_call_id) {
                    return Err(anyhow!("orphan tool result for id {tool_call_id}"));
                }
            }
        }
    }
    if !pending_tools.is_empty() {
        return Err(anyhow!("missing tool results for ids: {pending_tools:?}"));
    }
    Ok(())
}

pub fn persist_session_transcript(session_path: &Path, messages: &[ChatMessage]) -> Result<()> {
    for message in messages {
        crate::session::save_message(session_path, message)?;
    }
    Ok(())
}

pub fn rewrite_session_transcript(
    session_dir: &str,
    session_path: &mut PathBuf,
    messages: &[ChatMessage],
) -> Result<()> {
    *session_path = crate::session::new_session_path(session_dir);
    persist_session_transcript(session_path, messages)
}

fn non_system_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .filter(|m| !matches!(m, ChatMessage::System { .. }))
        .cloned()
        .collect()
}

fn split_into_turns(messages: &[ChatMessage]) -> Vec<Vec<ChatMessage>> {
    let mut turns = Vec::new();
    let mut current = Vec::new();
    for message in messages {
        if matches!(message, ChatMessage::User { .. }) && !current.is_empty() {
            turns.push(current);
            current = Vec::new();
        }
        current.push(message.clone());
    }
    if !current.is_empty() {
        turns.push(current);
    }
    turns
}

fn partition_for_keep(
    non_system: &[ChatMessage],
    keep_messages: usize,
) -> Option<(Vec<ChatMessage>, Vec<ChatMessage>)> {
    let turns = split_into_turns(non_system);
    if turns.len() <= 1 {
        return None;
    }

    let mut recent_turns = Vec::new();
    let mut kept_messages = 0usize;
    for turn in turns.iter().rev() {
        if !recent_turns.is_empty() && kept_messages >= keep_messages {
            break;
        }
        kept_messages += turn.len();
        recent_turns.push(turn.clone());
    }
    recent_turns.reverse();

    let kept_turns = recent_turns.len();
    if kept_turns == turns.len() {
        return None;
    }

    let older: Vec<ChatMessage> = turns[..turns.len() - kept_turns]
        .iter()
        .flatten()
        .cloned()
        .collect();
    let recent: Vec<ChatMessage> = recent_turns.iter().flatten().cloned().collect();
    validate_transcript(&recent).ok()?;
    Some((older, recent))
}

fn transcript_json_bytes(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| serde_json::to_vec(m).map(|v| v.len()).unwrap_or(0))
        .sum()
}

fn shrink_tool_messages(messages: &mut [ChatMessage]) {
    for message in messages.iter_mut() {
        if let ChatMessage::Tool { content, .. } = message {
            if content.chars().count() > TIER2_TOOL_MAX_CHARS {
                let trimmed: String = content.chars().take(TIER2_TOOL_MAX_CHARS).collect();
                *content = format!("{trimmed}…[trimmed for context]");
            }
        }
    }
}

fn append_summary(existing: Option<&str>, addition: &str) -> String {
    match existing.map(str::trim).filter(|s| !s.is_empty()) {
        Some(existing) => format!("{existing}\n{addition}"),
        None => addition.to_string(),
    }
}

fn build_compacted_messages(
    system_prompt: &str,
    summary: Option<&str>,
    mut recent: Vec<ChatMessage>,
) -> Result<Vec<ChatMessage>> {
    shrink_tool_messages(&mut recent);
    validate_transcript(&recent)?;
    let mut out = vec![ChatMessage::System {
        content: merge_system_prompt(system_prompt, summary),
    }];
    out.extend(recent);
    Ok(out)
}

pub async fn summarize_transcript(
    http: &Client,
    backend: &BackendDescriptor,
    model: &str,
    older: &[ChatMessage],
) -> Result<String> {
    let transcript = serde_json::to_string(older)?;
    let messages = vec![
        ChatMessage::System {
            content: "Summarize this Small Harness conversation for continuing context. Preserve goals, decisions, files touched, errors, and pending work. Be concise.".into(),
        },
        ChatMessage::User {
            content: transcript,
        },
    ];
    let req = ChatRequest {
        model,
        messages: &messages,
        tools: None,
        stream: true,
        stream_options: Some(StreamOptions {
            include_usage: false,
        }),
        max_tokens: None,
    };
    let mut out = String::new();
    stream_chat(http, backend, &req, None, |chunk| {
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                out.push_str(content);
            }
        }
    })
    .await?;
    Ok(out)
}

fn compact_notice(
    before_messages: usize,
    after_messages: usize,
    before_ratio: f64,
    after_ratio: f64,
    method: CompactMethod,
) -> String {
    let method_label = match method {
        CompactMethod::LlmSummary => "summarized",
        CompactMethod::DeterministicTrim => "trimmed",
        CompactMethod::None => "compacted",
    };
    format!(
        "Compacted {before_messages} messages → {after_messages} ({method_label}), budget {:.0}% → {:.0}%",
        before_ratio * 100.0,
        after_ratio * 100.0
    )
}

async fn compact_messages_core(req: CompactRequest<'_>) -> Result<CompactResult> {
    let CompactRequest {
        messages,
        system_prompt,
        tool_defs,
        keep_messages,
        limit_bytes,
        summarize_budget_bytes,
        http,
        backend,
        model,
        conversation_summary,
    } = req;

    let budget_before = measure_prompt_budget(system_prompt, messages, tool_defs);
    let before_messages = messages.len();
    let before_ratio = usage_ratio(&budget_before, limit_bytes);

    if before_messages <= keep_messages + 1 {
        return Ok(CompactResult {
            compacted: false,
            before_messages,
            after_messages: before_messages,
            before_ratio,
            after_ratio: before_ratio,
            method: CompactMethod::None,
            conversation_summary: conversation_summary.map(str::to_string),
        });
    }

    let non_system = non_system_messages(messages);
    let Some((older, recent)) = partition_for_keep(&non_system, keep_messages) else {
        return Ok(CompactResult {
            compacted: false,
            before_messages,
            after_messages: before_messages,
            before_ratio,
            after_ratio: before_ratio,
            method: CompactMethod::None,
            conversation_summary: conversation_summary.map(str::to_string),
        });
    };

    let use_tier2 = transcript_json_bytes(&older) > summarize_budget_bytes;
    let (method, new_summary) = if use_tier2 {
        (
            CompactMethod::DeterministicTrim,
            append_summary(
                conversation_summary,
                "[Earlier conversation trimmed to fit context budget]",
            ),
        )
    } else {
        let summary = summarize_transcript(http, backend, model, &older).await?;
        (CompactMethod::LlmSummary, summary.trim().to_string())
    };

    let compacted_messages = build_compacted_messages(system_prompt, Some(&new_summary), recent)?;
    *messages = compacted_messages;

    let budget_after = measure_prompt_budget(system_prompt, messages, tool_defs);
    let after_ratio = usage_ratio(&budget_after, limit_bytes);

    Ok(CompactResult {
        compacted: true,
        before_messages,
        after_messages: messages.len(),
        before_ratio,
        after_ratio,
        method,
        conversation_summary: Some(new_summary),
    })
}

pub async fn compact_messages(
    ctx: &mut CompactSessionContext<'_>,
    keep: Option<usize>,
) -> Result<CompactResult> {
    let guard = guard_config_from(ctx.config, ctx.model, ctx.is_local);
    let keep = keep.unwrap_or(guard.keep_messages);
    let summarize_budget =
        (guard.effective_limit_bytes as f64 * SUMMARIZE_BUDGET_FRACTION) as usize;

    compact_messages_core(CompactRequest {
        messages: ctx.messages,
        system_prompt: ctx.system_prompt,
        tool_defs: ctx.tool_defs,
        keep_messages: keep,
        limit_bytes: guard.effective_limit_bytes,
        summarize_budget_bytes: summarize_budget,
        http: ctx.http,
        backend: ctx.backend,
        model: ctx.model,
        conversation_summary: ctx.conversation_summary,
    })
    .await
}

pub async fn compact_session(
    ctx: &mut CompactSessionContext<'_>,
    session_dir: &str,
    session_path: &mut PathBuf,
    keep: Option<usize>,
) -> Result<CompactResult> {
    let result = compact_messages(ctx, keep).await?;

    if result.compacted {
        rewrite_session_transcript(session_dir, session_path, ctx.messages)?;
    }

    Ok(result)
}

pub async fn maybe_auto_compact(
    ctx: &mut CompactSessionContext<'_>,
    session_dir: &str,
    session_path: &mut PathBuf,
) -> Result<Option<CompactNotice>> {
    let guard = guard_config_from(ctx.config, ctx.model, ctx.is_local);
    let budget = measure_prompt_budget(ctx.system_prompt, ctx.messages, ctx.tool_defs);

    if !guard.auto_compact {
        if should_compact(
            &budget,
            guard.effective_limit_bytes,
            guard.compact_threshold,
        ) {
            return Ok(Some(CompactNotice {
                line: format!(
                    "  \x1b[33m!\x1b[0m \x1b[2mprompt budget is {} — run /compact or enable autoCompact\x1b[0m",
                    format_usage_line(&budget, guard.effective_limit_bytes)
                ),
                conversation_summary: ctx.conversation_summary.map(str::to_string),
                transcript_rewritten: false,
            }));
        }
        return Ok(None);
    }

    if !should_compact(
        &budget,
        guard.effective_limit_bytes,
        guard.compact_threshold,
    ) {
        return Ok(None);
    }

    let result = compact_messages(ctx, None).await?;

    if !result.compacted {
        return Ok(None);
    }

    rewrite_session_transcript(session_dir, session_path, ctx.messages)?;

    Ok(Some(CompactNotice {
        line: format!(
            "  \x1b[32m✓\x1b[0m \x1b[2m{}\x1b[0m",
            compact_notice(
                result.before_messages,
                result.after_messages,
                result.before_ratio,
                result.after_ratio,
                result.method
            )
        ),
        conversation_summary: result.conversation_summary.clone(),
        transcript_rewritten: true,
    }))
}

pub async fn maybe_compact_messages(
    messages: &mut Vec<ChatMessage>,
    system_prompt: &str,
    tool_defs: &[ToolDef],
    guard: &ContextGuardParams,
    http: &Client,
    backend: &BackendDescriptor,
    model: &str,
) -> Result<Option<CompactNotice>> {
    if !guard.auto_compact {
        return Ok(None);
    }

    let budget = measure_prompt_budget(system_prompt, messages, tool_defs);
    if !should_compact(
        &budget,
        guard.effective_limit_bytes,
        guard.compact_threshold,
    ) {
        return Ok(None);
    }

    let result = compact_messages_core(CompactRequest {
        messages,
        system_prompt,
        tool_defs,
        keep_messages: guard.keep_messages,
        limit_bytes: guard.effective_limit_bytes,
        summarize_budget_bytes: guard.summarize_budget_bytes,
        http,
        backend,
        model,
        conversation_summary: guard.conversation_summary.as_deref(),
    })
    .await?;

    if !result.compacted {
        return Ok(None);
    }

    Ok(Some(CompactNotice {
        line: format!(
            "  \x1b[32m✓\x1b[0m \x1b[2m{}\x1b[0m",
            compact_notice(
                result.before_messages,
                result.after_messages,
                result.before_ratio,
                result.after_ratio,
                result.method
            )
        ),
        conversation_summary: result.conversation_summary.clone(),
        transcript_rewritten: true,
    }))
}

pub fn context_status_lines(
    config: &AgentConfig,
    model: &str,
    is_local: bool,
    budget: &PromptBudget,
    last_notice: Option<&str>,
    conversation_summary: Option<&str>,
) -> Vec<String> {
    let guard = guard_config_from(config, model, is_local);
    let ratio = usage_ratio(budget, guard.effective_limit_bytes);
    let mut lines = vec![
        format!(
            "  \x1b[2meffectiveLimit\x1b[0m  {} (~{} model tokens, {:.0}% used, {} headroom)",
            format_bytes(guard.effective_limit_bytes),
            guard.model_context_tokens,
            ratio * 100.0,
            format_bytes(headroom_bytes(budget, guard.effective_limit_bytes))
        ),
        format!(
            "  \x1b[2mautoGuard\x1b[0m     autoCompact={} threshold={:.0}% reserve={:.0}%",
            guard.auto_compact,
            guard.compact_threshold * 100.0,
            config.context.reserve_ratio * 100.0
        ),
    ];
    if conversation_summary
        .map(str::trim)
        .is_some_and(|s| !s.is_empty())
    {
        lines.push("  \x1b[2msummary\x1b[0m       stored (persists across turns)".into());
    }
    if let Some(notice) = last_notice {
        lines.push(format!("  \x1b[2mlastGuard\x1b[0m     {notice}"));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ContextConfig;
    use crate::openai::{ToolCall, ToolFunction};

    fn sample_budget(total: usize) -> PromptBudget {
        PromptBudget {
            system_bytes: 0,
            transcript_bytes: 0,
            tool_schema_bytes: 0,
            tool_result_bytes: 0,
            total_bytes: total,
            effective_total_bytes: total,
            estimated_tokens: total / 4,
        }
    }

    #[test]
    fn effective_limit_uses_model_tokens_and_max_bytes_min() {
        let config = ContextConfig {
            model_context_tokens: Some(8192),
            max_bytes: Some(256 * 1024),
            reserve_ratio: 0.25,
            ..Default::default()
        };
        let limit = effective_limit_bytes(&config, "qwen2.5-coder:7b", true);
        assert_eq!(limit, 8192 * 4 * 3 / 4);
    }

    #[test]
    fn effective_limit_respects_max_bytes_cap() {
        let config = ContextConfig {
            model_context_tokens: Some(32768),
            max_bytes: Some(32 * 1024),
            ..Default::default()
        };
        let limit = effective_limit_bytes(&config, "big-model", true);
        assert_eq!(limit, 32 * 1024);
    }

    #[test]
    fn should_compact_at_threshold_not_before() {
        let budget = sample_budget(840);
        assert!(!should_compact(&budget, 1000, 0.85));
        assert!(should_compact(&sample_budget(850), 1000, 0.85));
    }

    #[test]
    fn auto_compact_defaults_local_on_cloud_off() {
        let config = ContextConfig::default();
        assert!(resolve_auto_compact(&config, true));
        assert!(!resolve_auto_compact(&config, false));
    }

    #[test]
    fn partition_keeps_tool_rounds_intact() {
        let messages = vec![
            ChatMessage::User {
                content: "one".into(),
            },
            ChatMessage::Assistant {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    kind: "function".into(),
                    function: ToolFunction {
                        name: "grep".into(),
                        arguments: "{}".into(),
                    },
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "call-1".into(),
                content: "match".into(),
            },
            ChatMessage::User {
                content: "two".into(),
            },
            ChatMessage::Assistant {
                content: Some("done".into()),
                tool_calls: vec![],
            },
        ];
        let (older, recent) = partition_for_keep(&messages, 2).expect("partition");
        assert_eq!(older.len(), 3);
        assert_eq!(recent.len(), 2);
        validate_transcript(&recent).expect("valid recent transcript");
    }

    #[test]
    fn validate_rejects_orphan_tool_message() {
        let messages = vec![ChatMessage::Tool {
            tool_call_id: "call-1".into(),
            content: "orphan".into(),
        }];
        assert!(validate_transcript(&messages).is_err());
    }

    #[test]
    fn merge_and_extract_summary_round_trip() {
        let merged = merge_system_prompt("base prompt", Some("kept goals"));
        assert!(merged.contains("kept goals"));
        assert_eq!(
            extract_conversation_summary(&merged).as_deref(),
            Some("kept goals")
        );
        assert_eq!(strip_conversation_summary(&merged), "base prompt");
    }

    #[test]
    fn default_model_tokens_conservative_for_unknown_local() {
        assert_eq!(default_model_context_tokens("my-model", true), 8192);
        assert_eq!(
            default_model_context_tokens("qwen2.5-coder:7b", true),
            32768
        );
        assert_eq!(default_model_context_tokens("gpt-4", false), 128_000);
    }
}
