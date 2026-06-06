//! Spec expansion for `/plan`.
//!
//! Takes a one- or two-sentence intent and expands it into an ambitious product
//! spec written to `.small-harness/spec.md`. Mirrors the drafting shape in
//! [`crate::handoff`]: a system prompt fixes the section contract, the model
//! drafts, [`ensure_spec_sections`] normalizes the result, and
//! [`render_fallback_spec`] provides a deterministic spec when the model draft
//! is empty or fails.
//!
//! The planner deliberately stays at the level of *what* and *why*, not *how*:
//! the system prompt forbids prescribing files, APIs, or code, because premature
//! technical detail in a spec cascades into downstream implementation errors.

use chrono::Utc;
use std::path::{Path, PathBuf};

/// Top-level sections every spec must contain, in order.
const SPEC_SECTIONS: &[&str] = &[
    "Goal",
    "User Outcomes",
    "Scope",
    "Out of Scope",
    "Done Criteria",
    "Open Questions",
];

/// Where `/plan` writes by default: `<workspace>/.small-harness/spec.md`.
/// Mirrors how the project prompt is rooted in `project_memory::load_project_prompt`.
pub fn default_spec_path(workspace_root: &str) -> PathBuf {
    Path::new(workspace_root)
        .join(".small-harness")
        .join("spec.md")
}

/// System prompt: fixes the section contract and forbids premature technical
/// detail (the "cascading errors" warning from the harness-design article).
pub fn planner_system_prompt() -> String {
    [
        "You expand a short product intent into a clear, ambitious specification for a software feature.",
        "Be ambitious about scope and user outcomes — but DO NOT over-specify implementation.",
        "Never prescribe file layouts, module names, function signatures, data structures, or code.",
        "Premature technical detail cascades into downstream errors: describe WHAT and WHY, not HOW.",
        "Return Markdown with exactly these top-level sections, in this order:",
        "## Goal",
        "## User Outcomes",
        "## Scope",
        "## Out of Scope",
        "## Done Criteria",
        "## Open Questions",
        "Goal: one or two sentences capturing the intent.",
        "User Outcomes: concrete, user-facing results, as bullets.",
        "Scope: what this feature includes, as bullets.",
        "Out of Scope: explicit non-goals, as bullets.",
        "Done Criteria: observable, testable conditions that mean the work is complete, as bullets.",
        "Open Questions: unknowns or decisions to confirm before building, as bullets. If none, write `None.`",
    ]
    .join("\n")
}

/// The user message: the intent plus the drafting rules.
pub fn render_planner_prompt(intent: &str) -> String {
    let mut out = String::new();
    out.push_str("Expand the following intent into a product spec.\n\n");
    out.push_str("Rules:\n");
    out.push_str("- Be ambitious about outcomes; keep the spec focused on deliverables, not implementation.\n");
    out.push_str("- Do not invent specific files, APIs, commands, or code.\n");
    out.push_str("- Prefer concrete, observable, testable Done Criteria.\n\n");
    out.push_str("Intent:\n");
    out.push_str(intent.trim());
    out
}

/// Normalize a model draft so every required section is present. Missing
/// sections are appended with a placeholder, mirroring
/// `handoff::ensure_required_sections`.
pub fn ensure_spec_sections(markdown: &str) -> String {
    let mut out = markdown.trim().to_string();
    for section in SPEC_SECTIONS {
        if !has_heading(&out, section) {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            let placeholder = if *section == "Open Questions" {
                "None."
            } else {
                "To be defined."
            };
            out.push_str(&format!("## {section}\n\n{placeholder}"));
        }
    }
    out.push('\n');
    out
}

/// Case-insensitive heading match, ignoring leading `#`s. Local copy of the
/// helper in `handoff.rs` to keep the modules decoupled.
fn has_heading(markdown: &str, section: &str) -> bool {
    markdown.lines().any(|line| {
        let trimmed = line.trim();
        let without_hash = trimmed.trim_start_matches('#').trim();
        !without_hash.is_empty() && without_hash.eq_ignore_ascii_case(section)
    })
}

/// Deterministic spec used when the model draft is empty or the request fails.
/// Always contains every required section so downstream readers (and `/plan
/// show`) get a well-formed file regardless of backend behavior.
pub fn render_fallback_spec(intent: &str, error: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str("# Spec\n\n");
    out.push_str(&format!("Generated: {}\n\n", Utc::now().to_rfc3339()));
    if let Some(error) = error {
        out.push_str(&format!("Draft status: model draft failed: {error}\n\n"));
    }
    out.push_str("## Goal\n\n");
    let goal = intent.trim();
    if goal.is_empty() {
        out.push_str("To be defined.\n\n");
    } else {
        out.push_str(goal);
        out.push_str("\n\n");
    }
    out.push_str("## User Outcomes\n\nTo be defined.\n\n");
    out.push_str("## Scope\n\nTo be defined.\n\n");
    out.push_str("## Out of Scope\n\nTo be defined.\n\n");
    out.push_str("## Done Criteria\n\nTo be defined.\n\n");
    out.push_str("## Open Questions\n\nNone.\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_path_is_under_small_harness() {
        let path = default_spec_path("/tmp/project");
        assert!(path.ends_with(".small-harness/spec.md"));
        assert!(path.starts_with("/tmp/project"));
    }

    #[test]
    fn ensure_spec_sections_appends_missing_sections() {
        let normalized = ensure_spec_sections("## Goal\n\nShip a CSV export.");
        for section in SPEC_SECTIONS {
            assert!(
                normalized.contains(&format!("## {section}")),
                "missing section: {section}"
            );
        }
        // The supplied Goal content is preserved.
        assert!(normalized.contains("Ship a CSV export."));
        // Open Questions gets the `None.` placeholder, others `To be defined.`
        assert!(normalized.contains("## Open Questions\n\nNone."));
    }

    #[test]
    fn ensure_spec_sections_is_idempotent_and_case_insensitive() {
        let once = ensure_spec_sections("# goal\n\nx\n\n# OPEN QUESTIONS\n\nNone.");
        // Existing headings (any case / hash count) are not duplicated.
        assert_eq!(
            once.matches("Goal").count() + once.matches("goal").count(),
            1
        );
        assert_eq!(once.to_lowercase().matches("open questions").count(), 1);
    }

    #[test]
    fn fallback_spec_contains_all_sections_and_intent() {
        let spec = render_fallback_spec("add a CSV export command", Some("boom"));
        for section in SPEC_SECTIONS {
            assert!(
                spec.contains(&format!("## {section}")),
                "missing: {section}"
            );
        }
        assert!(spec.contains("add a CSV export command"));
        assert!(spec.contains("model draft failed: boom"));
    }

    #[test]
    fn fallback_spec_handles_empty_intent() {
        let spec = render_fallback_spec("   ", None);
        assert!(spec.contains("## Goal\n\nTo be defined."));
        assert!(!spec.contains("Draft status"));
    }

    #[test]
    fn planner_prompt_includes_intent_and_rules() {
        let prompt = render_planner_prompt("  build a dashboard  ");
        assert!(prompt.contains("build a dashboard"));
        assert!(prompt.contains("not implementation"));
    }
}
