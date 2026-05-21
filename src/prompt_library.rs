use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub name: String,
    pub description: String,
    pub content: String,
    pub variables: Vec<String>,
}

pub struct PromptLibrary {
    built_in: HashMap<String, PromptTemplate>,
}

impl PromptLibrary {
    pub fn new() -> Self {
        let mut built_in = HashMap::new();

        // Code review prompt
        built_in.insert(
            "code_review".to_string(),
            PromptTemplate {
                name: "code_review".to_string(),
                description: "Review code for best practices and potential issues".to_string(),
                content: r#"Please review the following code for:
- Best practices and idiomatic usage
- Potential bugs or edge cases
- Performance considerations
- Security vulnerabilities
- Code organization and readability
- Testing suggestions

Provide specific, actionable feedback with examples where appropriate."#
                    .to_string(),
                variables: vec![],
            },
        );

        // Debug prompt
        built_in.insert(
            "debug".to_string(),
            PromptTemplate {
                name: "debug".to_string(),
                description: "Help debug and fix errors".to_string(),
                content:
                    r#"I'm experiencing an issue with my code. Here's the error message and context:
{{error_message}}

{{code_context}}

Please help me:
1. Identify the root cause of the error
2. Explain why this error is occurring
3. Provide a fix for the issue
4. Suggest how to prevent similar errors in the future"#
                        .to_string(),
                variables: vec!["error_message".to_string(), "code_context".to_string()],
            },
        );

        // Refactor prompt
        built_in.insert(
            "refactor".to_string(),
            PromptTemplate {
                name: "refactor".to_string(),
                description: "Suggest code refactoring improvements".to_string(),
                content: r#"Please review the following code and suggest refactoring improvements:
{{code}}

Focus on:
- Code duplication and DRY principles
- Function complexity and single responsibility
- Naming conventions
- Error handling
- Resource management
- Performance optimizations

Provide specific refactoring suggestions with before/after examples."#
                    .to_string(),
                variables: vec!["code".to_string()],
            },
        );

        // Document prompt
        built_in.insert(
            "document".to_string(),
            PromptTemplate {
                name: "document".to_string(),
                description: "Generate documentation for code".to_string(),
                content: r#"Please generate comprehensive documentation for the following code:
{{code}}

Include:
- High-level overview of what the code does
- Function/method signatures with parameter descriptions
- Return value descriptions
- Usage examples
- Edge cases and error conditions
- Dependencies and requirements
- Any important implementation notes"#
                    .to_string(),
                variables: vec!["code".to_string()],
            },
        );

        // Explain prompt
        built_in.insert("explain".to_string(), PromptTemplate {
            name: "explain".to_string(),
            description: "Explain code or concepts".to_string(),
            content: r#"Please explain the following code/concept in detail:
{{content}}

Break down:
- The overall purpose and approach
- Key components and their roles
- How different parts interact
- Important patterns or algorithms used
- Any non-obvious implementation details
- Context for when this approach is appropriate

Assume I have intermediate programming knowledge but may not be familiar with this specific domain."#.to_string(),
            variables: vec!["content".to_string()],
        });

        // Test prompt
        built_in.insert(
            "test".to_string(),
            PromptTemplate {
                name: "test".to_string(),
                description: "Generate unit tests for code".to_string(),
                content: r#"Please generate comprehensive unit tests for the following code:
{{code}}

Include tests for:
- Happy path / normal operation
- Edge cases and boundary conditions
- Error handling and invalid inputs
- Performance characteristics if relevant
- Integration scenarios if applicable

Use the appropriate testing framework for the language and follow testing best practices."#
                    .to_string(),
                variables: vec!["code".to_string()],
            },
        );

        PromptLibrary { built_in }
    }

    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.built_in.get(name)
    }

    pub fn list(&self) -> Vec<&PromptTemplate> {
        let mut out = self.built_in.values().collect::<Vec<_>>();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    pub fn render(&self, name: &str, variables: &HashMap<String, String>) -> Result<String> {
        let template = self
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Template '{}' not found", name))?;

        let mut content = template.content.clone();
        for (key, value) in variables {
            content = content.replace(&format!("{{{{{}}}}}", key), value);
        }

        Ok(content)
    }
}

impl Default for PromptLibrary {
    fn default() -> Self {
        Self::new()
    }
}

pub fn save_prompt(session_dir: &str, name: &str, content: &str) -> Result<()> {
    let prompt_dir = Path::new(session_dir).join("prompts");
    fs::create_dir_all(&prompt_dir)?;

    let prompt_path = prompt_path(&prompt_dir, name)?;
    fs::write(&prompt_path, content)?;

    Ok(())
}

pub fn load_prompt(session_dir: &str, name: &str) -> Result<String> {
    let prompt_dir = Path::new(session_dir).join("prompts");
    let prompt_path = prompt_path(&prompt_dir, name)?;

    if !prompt_path.exists() {
        return Err(anyhow::anyhow!("Prompt '{}' not found", name));
    }

    let content = fs::read_to_string(&prompt_path)?;
    Ok(content)
}

pub fn list_prompts(session_dir: &str) -> Result<Vec<String>> {
    let prompt_dir = Path::new(session_dir).join("prompts");

    if !prompt_dir.exists() {
        return Ok(vec![]);
    }

    let mut prompts = Vec::new();
    for entry in fs::read_dir(&prompt_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "md") {
            if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                prompts.push(name.to_string());
            }
        }
    }

    prompts.sort();
    Ok(prompts)
}

pub fn delete_prompt(session_dir: &str, name: &str) -> Result<()> {
    let prompt_dir = Path::new(session_dir).join("prompts");
    let prompt_path = prompt_path(&prompt_dir, name)?;

    if !prompt_path.exists() {
        return Err(anyhow::anyhow!("Prompt '{}' not found", name));
    }

    fs::remove_file(&prompt_path)?;
    Ok(())
}

pub fn export_prompts(session_dir: &str, export_path: &Path) -> Result<()> {
    let prompt_dir = Path::new(session_dir).join("prompts");

    if !prompt_dir.exists() {
        return Err(anyhow::anyhow!("No prompts directory found"));
    }

    let prompts = list_prompts(session_dir)?;
    let mut export_data = HashMap::new();

    for prompt_name in &prompts {
        let content = load_prompt(session_dir, prompt_name)?;
        export_data.insert(prompt_name.clone(), content);
    }

    let json = serde_json::to_string_pretty(&export_data)?;
    if let Some(parent) = export_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(export_path, json)?;

    Ok(())
}

pub fn import_prompts(session_dir: &str, import_path: &Path) -> Result<usize> {
    if !import_path.exists() {
        return Err(anyhow::anyhow!("Import file not found"));
    }

    let json = fs::read_to_string(import_path)?;
    let import_data: HashMap<String, String> = serde_json::from_str(&json)?;

    let prompt_dir = Path::new(session_dir).join("prompts");
    fs::create_dir_all(&prompt_dir)?;

    let mut count = 0;
    for (name, content) in import_data {
        let prompt_path = prompt_path(&prompt_dir, &name)?;
        fs::write(&prompt_path, content)?;
        count += 1;
    }

    Ok(count)
}

fn prompt_path(prompt_dir: &Path, name: &str) -> Result<std::path::PathBuf> {
    validate_prompt_name(name)?;
    Ok(prompt_dir.join(format!("{name}.md")))
}

pub fn validate_prompt_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(anyhow::anyhow!("Prompt name cannot be empty"));
    }
    if name.starts_with('.') || name.contains("..") {
        return Err(anyhow::anyhow!(
            "Prompt name cannot contain hidden or parent path segments"
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        return Err(anyhow::anyhow!(
            "Prompt name can only contain letters, numbers, '.', '_' and '-'"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_prompt_names_that_escape_directory() {
        assert!(validate_prompt_name("../escape").is_err());
        assert!(validate_prompt_name("nested/name").is_err());
        assert!(validate_prompt_name(".hidden").is_err());
        assert!(validate_prompt_name("safe-name_1").is_ok());
    }

    #[test]
    fn imports_and_exports_prompts_round_trip() {
        let source = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();
        save_prompt(source.path().to_str().unwrap(), "review", "review prompt").unwrap();
        let export_path = source.path().join("exports/prompts.json");

        export_prompts(source.path().to_str().unwrap(), &export_path).unwrap();
        let count = import_prompts(dest.path().to_str().unwrap(), &export_path).unwrap();

        assert_eq!(count, 1);
        assert_eq!(
            load_prompt(dest.path().to_str().unwrap(), "review").unwrap(),
            "review prompt"
        );
    }
}
