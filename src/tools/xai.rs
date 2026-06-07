use async_trait::async_trait;
use serde_json::{json, Value};

use super::{image_base64_for_data_url, PathPolicy, Tool, ToolPreview};
use crate::backends::{backend, BackendName};

#[derive(Debug, Clone, Copy)]
pub enum XaiToolKind {
    GenerateText,
    MultiAgent,
    WebSearch,
    XSearch,
    CodeExecution,
    GenerateImage,
    Critique,
    AnalyzeImage,
    DeepResearch,
}

#[derive(Clone)]
pub struct XaiTool {
    pub kind: XaiToolKind,
    pub http: reqwest::Client,
    pub path_policy: PathPolicy,
}

const TEXT_MODEL: &str = "grok-4.3";
const MULTI_AGENT_MODEL: &str = "grok-4.20-multi-agent-0309";
const IMAGE_MODEL: &str = "grok-imagine-image-quality";
const RESPONSE_TIMEOUT_SECS: u64 = 120;
const IMAGE_TIMEOUT_SECS: u64 = 180;

impl XaiToolKind {
    fn name(self) -> &'static str {
        match self {
            Self::GenerateText => "xai_generate_text",
            Self::MultiAgent => "xai_multi_agent",
            Self::WebSearch => "xai_web_search",
            Self::XSearch => "xai_x_search",
            Self::CodeExecution => "xai_code_execution",
            Self::GenerateImage => "xai_generate_image",
            Self::Critique => "xai_critique",
            Self::AnalyzeImage => "xai_analyze_image",
            Self::DeepResearch => "xai_deep_research",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::GenerateText => "Generate text with Grok/xAI using XAI_API_KEY or /login xai OAuth credentials.",
            Self::MultiAgent => "Run xAI/Grok multi-agent research with server-side web and X search enabled.",
            Self::WebSearch => "Ask Grok to search the web using xAI's server-side web_search tool and return a cited answer.",
            Self::XSearch => "Ask Grok to search X/Twitter using xAI's server-side x_search tool and return a cited answer.",
            Self::CodeExecution => "Ask Grok to execute or analyze Python code using xAI's server-side code_interpreter tool.",
            Self::GenerateImage => "Generate or edit an image with xAI Grok Imagine. Returns temporary image URLs or base64 if requested.",
            Self::Critique => "Ask Grok for a structured critique of code, writing, design, logic, security, or performance.",
            Self::AnalyzeImage => "Ask Grok to analyze an image URL, data URI, or local image file.",
            Self::DeepResearch => "Run thorough multi-step Grok research with server-side web, X search, and code execution tools.",
        }
    }
}

#[async_trait]
impl Tool for XaiTool {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn description(&self) -> &'static str {
        self.kind.description()
    }

    fn input_schema(&self) -> Value {
        schema_for(self.kind)
    }

    fn require_approval(&self, _args: &Value) -> bool {
        // These tools perform cloud calls, may consume paid xAI quota, and some
        // variants use server-side web/X/code tools. Gate them consistently.
        true
    }

    async fn preview(&self, args: &Value) -> Option<ToolPreview> {
        let subject = args
            .get("prompt")
            .or_else(|| args.get("query"))
            .or_else(|| args.get("topic"))
            .or_else(|| args.get("code"))
            .or_else(|| args.get("content"))
            .or_else(|| args.get("image"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let subject = if subject.chars().count() > 100 {
            format!("{}…", subject.chars().take(100).collect::<String>())
        } else {
            subject.to_string()
        };
        Some(ToolPreview {
            summary: format!(
                "Call {}{}",
                self.name(),
                if subject.is_empty() {
                    String::new()
                } else {
                    format!(" for {subject:?}")
                }
            ),
            diff: None,
            risk: Some("Cloud xAI request; may use paid quota and send prompt/data to xAI.".into()),
        })
    }

    async fn execute(&self, args: Value) -> Value {
        let result = match self.kind {
            XaiToolKind::GenerateText => self.generate_text(args).await,
            XaiToolKind::MultiAgent => self.multi_agent(args).await,
            XaiToolKind::WebSearch => self.web_search(args).await,
            XaiToolKind::XSearch => self.x_search(args).await,
            XaiToolKind::CodeExecution => self.code_execution(args).await,
            XaiToolKind::GenerateImage => self.generate_image(args).await,
            XaiToolKind::Critique => self.critique(args).await,
            XaiToolKind::AnalyzeImage => self.analyze_image(args).await,
            XaiToolKind::DeepResearch => self.deep_research(args).await,
        };
        result.unwrap_or_else(|e| json!({ "error": e }))
    }
}

impl XaiTool {
    async fn generate_text(&self, args: Value) -> Result<Value, String> {
        let prompt = required_str(&args, &["prompt", "text", "query"])?;
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| TEXT_MODEL.into());
        let system = optional_str(&args, &["system", "instructions"]);
        let body = responses_body(
            &model,
            user_text_input(&prompt),
            system,
            Vec::new(),
            max_tokens(&args),
        );
        self.responses_result(body).await
    }

    async fn web_search(&self, args: Value) -> Result<Value, String> {
        let query = required_str(&args, &["query", "prompt", "search_term"])?;
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| TEXT_MODEL.into());
        let mut tool = json!({ "type": "web_search" });
        let mut filters = serde_json::Map::new();
        if let Some(domains) =
            string_array(&args, &["allowed_domains", "domains"]).filter(|values| !values.is_empty())
        {
            filters.insert("allowed_domains".into(), json!(domains));
        }
        if let Some(domains) = string_array(&args, &["excluded_domains", "exclude_domains"])
            .filter(|values| !values.is_empty())
        {
            filters.insert("excluded_domains".into(), json!(domains));
        }
        if !filters.is_empty() {
            tool["filters"] = Value::Object(filters);
        }
        copy_bool(
            &mut tool,
            &args,
            "enable_image_understanding",
            &["enable_image_understanding", "image_understanding"],
        );
        copy_bool(
            &mut tool,
            &args,
            "enable_image_search",
            &["enable_image_search", "image_search"],
        );
        let prompt =
            format!("Search the web and answer with citations where available. Query: {query}");
        let body = responses_body(
            &model,
            user_text_input(&prompt),
            None,
            vec![tool],
            max_tokens(&args),
        );
        self.responses_result(body).await
    }

    async fn x_search(&self, args: Value) -> Result<Value, String> {
        let query = required_str(&args, &["query", "prompt", "search_term"])?;
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| TEXT_MODEL.into());
        let mut tool = json!({ "type": "x_search" });
        copy_string_array(
            &mut tool,
            &args,
            "allowed_x_handles",
            &["allowed_x_handles", "handles", "from_handles"],
        );
        copy_string_array(
            &mut tool,
            &args,
            "excluded_x_handles",
            &["excluded_x_handles", "exclude_handles"],
        );
        copy_string(&mut tool, &args, "from_date", &["from_date", "since"]);
        copy_string(&mut tool, &args, "to_date", &["to_date", "until"]);
        copy_bool(
            &mut tool,
            &args,
            "enable_image_understanding",
            &["enable_image_understanding", "image_understanding"],
        );
        copy_bool(
            &mut tool,
            &args,
            "enable_video_understanding",
            &["enable_video_understanding", "video_understanding"],
        );
        let mut prompt =
            format!("Search X/Twitter and answer with citations where available. Query: {query}");
        if let Some(count) = args.get("count").and_then(Value::as_u64) {
            prompt.push_str(&format!(
                "\nReturn at most {count} especially relevant posts/items when possible."
            ));
        }
        let body = responses_body(
            &model,
            user_text_input(&prompt),
            None,
            vec![tool],
            max_tokens(&args),
        );
        self.responses_result(body).await
    }

    async fn code_execution(&self, args: Value) -> Result<Value, String> {
        let code = required_str(&args, &["code", "prompt", "query"])?;
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| TEXT_MODEL.into());
        let question = optional_str(&args, &["question", "instructions"])
            .unwrap_or_else(|| "Execute or analyze this Python code. If execution is useful, use the code interpreter and explain the result.".into());
        let prompt = format!("{question}\n\n```python\n{code}\n```");
        let body = responses_body(
            &model,
            user_text_input(&prompt),
            None,
            vec![json!({ "type": "code_interpreter" })],
            max_tokens(&args),
        );
        self.responses_result(body).await
    }

    async fn critique(&self, args: Value) -> Result<Value, String> {
        let content = required_str(&args, &["content", "text", "code"])?;
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| TEXT_MODEL.into());
        let aspect =
            optional_str(&args, &["aspect", "focus"]).unwrap_or_else(|| "overall quality".into());
        let tone = optional_str(&args, &["tone"]).unwrap_or_else(|| "constructive".into());
        let prompt = format!(
            "Provide a {tone}, structured critique focused on {aspect}. Include strengths, weaknesses, concrete fixes, and any important risks.\n\nContent:\n{content}"
        );
        let body = responses_body(
            &model,
            user_text_input(&prompt),
            None,
            Vec::new(),
            max_tokens(&args),
        );
        self.responses_result(body).await
    }

    async fn multi_agent(&self, args: Value) -> Result<Value, String> {
        let query = required_str(&args, &["query", "prompt", "topic"])?;
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| MULTI_AGENT_MODEL.into());
        let mut prompt = format!("Research this using multiple specialized agents, then synthesize a concise, well-sourced answer:\n\n{query}");
        if let Some(n) = args.get("num_agents").and_then(Value::as_u64) {
            prompt.push_str(&format!("\nDesired agent count: {n}."));
        }
        if let Some(effort) = optional_str(&args, &["reasoning_effort", "effort"]) {
            prompt.push_str(&format!("\nReasoning effort: {effort}."));
        }
        let tools = maybe_search_tools(&args, true, true, false);
        let body = responses_body(
            &model,
            user_text_input(&prompt),
            None,
            tools,
            max_tokens(&args),
        );
        self.responses_result(body).await
    }

    async fn deep_research(&self, args: Value) -> Result<Value, String> {
        let topic = required_str(&args, &["topic", "query", "prompt"])?;
        let depth =
            optional_str(&args, &["depth", "reasoning_effort"]).unwrap_or_else(|| "high".into());
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| MULTI_AGENT_MODEL.into());
        let prompt = format!(
            "Conduct {depth}-depth multi-step research on this topic. Search broadly, cross-check sources, use code execution for quantitative checks when useful, cite sources, and separate findings from uncertainty.\n\nTopic: {topic}"
        );
        let tools = maybe_search_tools(&args, true, true, true);
        let body = responses_body(
            &model,
            user_text_input(&prompt),
            None,
            tools,
            max_tokens(&args),
        );
        self.responses_result(body).await
    }

    async fn analyze_image(&self, args: Value) -> Result<Value, String> {
        let image = required_str(&args, &["image", "image_url", "url", "path"])?;
        let question = optional_str(&args, &["question", "prompt"]).unwrap_or_else(|| {
            "Describe this image in detail and answer any visually evident questions.".into()
        });
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| TEXT_MODEL.into());
        let image_url = self.image_input_to_url(&image).await?;
        let input = json!([{
            "role": "user",
            "content": [
                { "type": "input_text", "text": question },
                { "type": "input_image", "image_url": image_url, "detail": "auto" }
            ]
        }]);
        let body = responses_body(&model, input, None, Vec::new(), max_tokens(&args));
        self.responses_result(body).await
    }

    async fn generate_image(&self, args: Value) -> Result<Value, String> {
        let prompt = required_str(&args, &["prompt", "description"])?;
        let model = optional_str(&args, &["model"]).unwrap_or_else(|| IMAGE_MODEL.into());
        let backend = backend(BackendName::Xai);
        let token = xai_access_token(&self.http, &backend.api_key).await?;
        let source_image = optional_str(&args, &["image", "image_url", "url", "path"]);
        let mut body = json!({
            "model": model,
            "prompt": prompt,
            "n": args.get("n").and_then(Value::as_u64).unwrap_or(1).clamp(1, 10),
        });
        if let Some(format) = optional_str(&args, &["response_format", "image_format"]) {
            body["response_format"] = json!(format);
        }
        if let Some(aspect_ratio) = optional_str(&args, &["aspect_ratio", "aspectRatio"])
            .or_else(|| aspect_ratio_from_size(&args))
        {
            body["aspect_ratio"] = json!(aspect_ratio);
        }
        if let Some(resolution) = optional_str(&args, &["resolution"]) {
            body["resolution"] = json!(resolution);
        }
        let endpoint = if let Some(image) = source_image {
            let url = self.image_input_to_url(&image).await?;
            body["image"] = json!({ "type": "image_url", "url": url });
            "images/edits"
        } else {
            "images/generations"
        };
        let url = format!("{}/{}", backend.base_url.trim_end_matches('/'), endpoint);
        let resp = self
            .http
            .post(url)
            .bearer_auth(token)
            .header("content-type", "application/json")
            .timeout(std::time::Duration::from_secs(IMAGE_TIMEOUT_SECS))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("xAI image request failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("xAI image HTTP {status}: {}", body.trim()));
        }
        let raw: Value = resp
            .json()
            .await
            .map_err(|e| format!("invalid xAI image JSON: {e}"))?;
        Ok(json!({
            "data": raw.get("data").cloned().unwrap_or(Value::Null),
            "usage": raw.get("usage").cloned(),
            "raw": raw,
        }))
    }

    async fn responses_result(&self, body: Value) -> Result<Value, String> {
        let raw = post_xai_responses(&self.http, body).await?;
        let text = extract_output_text(&raw);
        Ok(json!({
            "text": text,
            "citations": raw.get("citations").cloned(),
            "usage": raw.get("usage").cloned(),
            "server_side_tool_usage": raw.get("server_side_tool_usage").cloned(),
            "tool_calls": raw.get("tool_calls").cloned(),
            "raw": raw,
        }))
    }

    async fn image_input_to_url(&self, image: &str) -> Result<String, String> {
        if image.starts_with("http://")
            || image.starts_with("https://")
            || image.starts_with("data:")
        {
            return Ok(image.to_string());
        }
        if let Some(error) = self.path_policy.deny_path(image) {
            return Err(error);
        }
        let resolved = self.path_policy.resolve(image);
        let mime = image_mime(&resolved.normalized.display().to_string()).ok_or_else(|| {
            "local image path must end in .png, .jpg, .jpeg, .gif, or .webp".to_string()
        })?;
        let bytes = tokio::fs::read(&resolved.normalized)
            .await
            .map_err(|e| format!("read image failed: {e}"))?;
        Ok(format!(
            "data:{mime};base64,{}",
            image_base64_for_data_url(&bytes)
        ))
    }
}

async fn post_xai_responses(client: &reqwest::Client, body: Value) -> Result<Value, String> {
    let backend = backend(BackendName::Xai);
    let token = xai_access_token(client, &backend.api_key).await?;
    let url = resolve_xai_responses_url(&backend.base_url);
    let resp = client
        .post(url)
        .bearer_auth(token)
        .header("content-type", "application/json")
        .timeout(std::time::Duration::from_secs(RESPONSE_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("xAI request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("xAI HTTP {status}: {}", body.trim()));
    }
    resp.json::<Value>()
        .await
        .map_err(|e| format!("invalid xAI JSON: {e}"))
}

async fn xai_access_token(client: &reqwest::Client, env_key: &str) -> Result<String, String> {
    if !env_key.is_empty() {
        Ok(env_key.to_string())
    } else {
        crate::xai_oauth::access_token(client)
            .await
            .map_err(|e| e.to_string())
    }
}

fn resolve_xai_responses_url(base_url: &str) -> String {
    let normalized = base_url.trim_end_matches('/');
    if normalized.ends_with("/responses") {
        normalized.to_string()
    } else {
        format!("{normalized}/responses")
    }
}

fn responses_body(
    model: &str,
    input: Value,
    instructions: Option<String>,
    tools: Vec<Value>,
    max_tokens: Option<u32>,
) -> Value {
    let model = crate::xai_responses::canonical_xai_model(model).unwrap_or(model);
    let mut body = json!({
        "model": model,
        "stream": false,
        "input": input,
    });
    if let Some(instructions) = instructions.filter(|s| !s.trim().is_empty()) {
        body["instructions"] = json!(instructions);
    }
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    if let Some(max_tokens) = max_tokens {
        body["max_output_tokens"] = json!(max_tokens);
    }
    body
}

fn user_text_input(prompt: &str) -> Value {
    json!([{ "role": "user", "content": prompt }])
}

fn required_str(args: &Value, keys: &[&str]) -> Result<String, String> {
    optional_str(args, keys).ok_or_else(|| format!("missing required argument: {}", keys[0]))
}

fn optional_str(args: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        args.get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
    })
}

fn max_tokens(args: &Value) -> Option<u32> {
    args.get("max_tokens")
        .or_else(|| args.get("maxTokens"))
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
}

fn string_array(args: &Value, keys: &[&str]) -> Option<Vec<String>> {
    keys.iter().find_map(|key| {
        let v = args.get(*key)?;
        if let Some(s) = v.as_str() {
            return Some(
                s.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .collect(),
            );
        }
        Some(
            v.as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .collect(),
        )
    })
}

fn copy_string(target: &mut Value, args: &Value, target_key: &str, source_keys: &[&str]) {
    if let Some(value) = optional_str(args, source_keys) {
        target[target_key] = json!(value);
    }
}

fn copy_bool(target: &mut Value, args: &Value, target_key: &str, source_keys: &[&str]) {
    for key in source_keys {
        if let Some(value) = args.get(*key).and_then(Value::as_bool) {
            target[target_key] = json!(value);
            return;
        }
    }
}

fn copy_string_array(target: &mut Value, args: &Value, target_key: &str, source_keys: &[&str]) {
    if let Some(values) = string_array(args, source_keys).filter(|values| !values.is_empty()) {
        target[target_key] = json!(values);
    }
}

fn maybe_search_tools(args: &Value, default_web: bool, default_x: bool, code: bool) -> Vec<Value> {
    let use_web = args
        .get("web_search")
        .and_then(Value::as_bool)
        .unwrap_or(default_web);
    let use_x = args
        .get("x_search")
        .and_then(Value::as_bool)
        .unwrap_or(default_x);
    let mut tools = Vec::new();
    if use_web {
        tools.push(json!({ "type": "web_search" }));
    }
    if use_x {
        tools.push(json!({ "type": "x_search" }));
    }
    if code {
        tools.push(json!({ "type": "code_interpreter" }));
    }
    tools
}

fn aspect_ratio_from_size(args: &Value) -> Option<String> {
    let size = optional_str(args, &["size"])?;
    let (w, h) = size.split_once('x')?;
    let w: u64 = w.trim().parse().ok()?;
    let h: u64 = h.trim().parse().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    let g = gcd(w, h);
    Some(format!("{}:{}", w / g, h / g))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn image_mime(path: &str) -> Option<&'static str> {
    let lower = path.to_lowercase();
    if lower.ends_with(".png") {
        Some("image/png")
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        Some("image/jpeg")
    } else if lower.ends_with(".gif") {
        Some("image/gif")
    } else if lower.ends_with(".webp") {
        Some("image/webp")
    } else {
        None
    }
}

fn extract_output_text(raw: &Value) -> String {
    if let Some(text) = raw.get("output_text").and_then(Value::as_str) {
        return text.to_string();
    }
    let mut parts = Vec::new();
    collect_text_parts(raw.get("output").unwrap_or(raw), &mut parts);
    parts.join("")
}

fn collect_text_parts(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_text_parts(value, parts);
            }
        }
        Value::Object(map) => {
            let ty = map.get("type").and_then(Value::as_str).unwrap_or_default();
            if matches!(ty, "output_text" | "text" | "summary_text") {
                if let Some(text) = map.get("text").and_then(Value::as_str) {
                    parts.push(text.to_string());
                    return;
                }
            }
            if let Some(content) = map.get("content") {
                collect_text_parts(content, parts);
            }
            if let Some(summary) = map.get("summary") {
                collect_text_parts(summary, parts);
            }
        }
        _ => {}
    }
}

fn schema_for(kind: XaiToolKind) -> Value {
    match kind {
        XaiToolKind::GenerateText => json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string", "description": "Prompt to send to Grok" },
                "model": { "type": "string", "description": "Optional xAI model (default grok-4.3)" },
                "system": { "type": "string", "description": "Optional system/developer instructions" },
                "max_tokens": { "type": "integer", "minimum": 1 }
            },
            "required": ["prompt"]
        }),
        XaiToolKind::WebSearch => json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "model": { "type": "string", "description": "Optional xAI model (default grok-4.3)" },
                "allowed_domains": { "type": "array", "items": { "type": "string" }, "maxItems": 5 },
                "excluded_domains": { "type": "array", "items": { "type": "string" }, "maxItems": 5 },
                "enable_image_understanding": { "type": "boolean" },
                "enable_image_search": { "type": "boolean" },
                "max_tokens": { "type": "integer", "minimum": 1 }
            },
            "required": ["query"]
        }),
        XaiToolKind::XSearch => json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "model": { "type": "string", "description": "Optional xAI model (default grok-4.3)" },
                "allowed_x_handles": { "type": "array", "items": { "type": "string" }, "maxItems": 20 },
                "excluded_x_handles": { "type": "array", "items": { "type": "string" }, "maxItems": 20 },
                "from_date": { "type": "string", "description": "YYYY-MM-DD" },
                "to_date": { "type": "string", "description": "YYYY-MM-DD" },
                "enable_image_understanding": { "type": "boolean" },
                "enable_video_understanding": { "type": "boolean" },
                "count": { "type": "integer", "minimum": 1, "maximum": 10 },
                "max_tokens": { "type": "integer", "minimum": 1 }
            },
            "required": ["query"]
        }),
        XaiToolKind::CodeExecution => json!({
            "type": "object",
            "properties": {
                "code": { "type": "string", "description": "Python code or data-analysis prompt" },
                "question": { "type": "string", "description": "Optional question/instructions for Grok" },
                "model": { "type": "string", "description": "Optional xAI model (default grok-4.3)" },
                "max_tokens": { "type": "integer", "minimum": 1 }
            },
            "required": ["code"]
        }),
        XaiToolKind::GenerateImage => json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string" },
                "model": { "type": "string", "description": "Image model (default grok-imagine-image-quality)" },
                "image": { "type": "string", "description": "Optional source image URL, data URI, or local path for editing" },
                "n": { "type": "integer", "minimum": 1, "maximum": 10 },
                "aspect_ratio": { "type": "string", "description": "e.g. 1:1, 16:9, 3:2" },
                "size": { "type": "string", "description": "Compatibility alias; converted to aspect ratio, e.g. 1024x1024" },
                "resolution": { "type": "string", "enum": ["1k", "2k"] },
                "response_format": { "type": "string", "enum": ["url", "b64_json"] }
            },
            "required": ["prompt"]
        }),
        XaiToolKind::Critique => json!({
            "type": "object",
            "properties": {
                "content": { "type": "string" },
                "aspect": { "type": "string", "description": "code, design, writing, logic, security, performance, etc." },
                "tone": { "type": "string", "description": "constructive, strict, balanced" },
                "model": { "type": "string", "description": "Optional xAI model (default grok-4.3)" },
                "max_tokens": { "type": "integer", "minimum": 1 }
            },
            "required": ["content"]
        }),
        XaiToolKind::AnalyzeImage => json!({
            "type": "object",
            "properties": {
                "image": { "type": "string", "description": "Image URL, data URI, or local path" },
                "question": { "type": "string", "description": "Question to answer about the image" },
                "model": { "type": "string", "description": "Optional vision-capable xAI model (default grok-4.3)" },
                "max_tokens": { "type": "integer", "minimum": 1 }
            },
            "required": ["image"]
        }),
        XaiToolKind::MultiAgent => json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "num_agents": { "type": "integer", "minimum": 1, "maximum": 16 },
                "reasoning_effort": { "type": "string", "enum": ["medium", "high"] },
                "model": { "type": "string", "description": "Optional xAI multi-agent model" },
                "web_search": { "type": "boolean" },
                "x_search": { "type": "boolean" },
                "max_tokens": { "type": "integer", "minimum": 1 }
            },
            "required": ["query"]
        }),
        XaiToolKind::DeepResearch => json!({
            "type": "object",
            "properties": {
                "topic": { "type": "string" },
                "depth": { "type": "string", "enum": ["low", "medium", "high"] },
                "model": { "type": "string", "description": "Optional xAI research model" },
                "web_search": { "type": "boolean" },
                "x_search": { "type": "boolean" },
                "max_tokens": { "type": "integer", "minimum": 1 }
            },
            "required": ["topic"]
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xai_tool_names_match_public_suite() {
        let names: Vec<&str> = [
            XaiToolKind::GenerateText,
            XaiToolKind::MultiAgent,
            XaiToolKind::WebSearch,
            XaiToolKind::XSearch,
            XaiToolKind::CodeExecution,
            XaiToolKind::GenerateImage,
            XaiToolKind::Critique,
            XaiToolKind::AnalyzeImage,
            XaiToolKind::DeepResearch,
        ]
        .into_iter()
        .map(XaiToolKind::name)
        .collect();
        assert_eq!(
            names,
            vec![
                "xai_generate_text",
                "xai_multi_agent",
                "xai_web_search",
                "xai_x_search",
                "xai_code_execution",
                "xai_generate_image",
                "xai_critique",
                "xai_analyze_image",
                "xai_deep_research",
            ]
        );
    }

    #[test]
    fn extracts_responses_output_text() {
        let raw = json!({
            "output": [{
                "type": "message",
                "content": [
                    { "type": "output_text", "text": "hello" },
                    { "type": "output_text", "text": " world" }
                ]
            }]
        });
        assert_eq!(extract_output_text(&raw), "hello world");
    }

    #[test]
    fn size_converts_to_aspect_ratio() {
        assert_eq!(
            aspect_ratio_from_size(&json!({"size":"1792x1024"})).as_deref(),
            Some("7:4")
        );
        assert_eq!(
            aspect_ratio_from_size(&json!({"size":"1024x1024"})).as_deref(),
            Some("1:1")
        );
    }

    #[test]
    fn web_search_schema_requires_query() {
        let schema = schema_for(XaiToolKind::WebSearch);
        assert_eq!(schema["required"], json!(["query"]));
        assert!(schema["properties"].get("allowed_domains").is_some());
    }
}
