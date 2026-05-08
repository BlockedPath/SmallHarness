use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::AgentConfig;
use crate::project_memory::load_project_index;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchEditOperation {
    pub file_path: String,
    pub operation: EditOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum EditOperation {
    Replace {
        old_string: String,
        new_string: String,
    },
    Insert {
        position: InsertPosition,
        content: String,
    },
    Delete {
        pattern: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InsertPosition {
    AfterLine { line: usize },
    BeforeLine { line: usize },
    AtEnd,
    AtStart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchPreview {
    pub operations: Vec<BatchEditOperation>,
    pub total_files: usize,
    pub estimated_changes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchResult {
    pub successful: Vec<String>,
    pub failed: Vec<FailedOperation>,
    pub skipped: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailedOperation {
    pub file_path: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrossFileReference {
    pub from_file: String,
    pub to_file: String,
    pub reference_type: String,
    pub line: usize,
}

pub fn find_cross_file_references(
    config: &AgentConfig,
    target_file: &str,
) -> Result<Vec<CrossFileReference>> {
    let Some(index) = load_project_index(config)? else {
        return Ok(Vec::new());
    };

    let target_symbols: HashMap<String, usize> = index
        .files
        .iter()
        .find(|f| f.path == target_file)
        .map(|f| {
            f.symbols
                .iter()
                .map(|s| (s.name.clone(), s.line))
                .collect()
        })
        .unwrap_or_default();

    let mut references = Vec::new();

    for file in &index.files {
        if file.path == target_file {
            continue;
        }

        for import in &file.imports {
            if target_symbols.contains_key(import) || import.contains(target_file) {
                references.push(CrossFileReference {
                    from_file: file.path.clone(),
                    to_file: target_file.to_string(),
                    reference_type: "import".to_string(),
                    line: 0,
                });
            }
        }

        for symbol in &file.symbols {
            if target_symbols.contains_key(&symbol.name) {
                references.push(CrossFileReference {
                    from_file: file.path.clone(),
                    to_file: target_file.to_string(),
                    reference_type: "symbol".to_string(),
                    line: symbol.line,
                });
            }
        }
    }

    Ok(references)
}

pub fn find_related_files(config: &AgentConfig, file_path: &str) -> Result<Vec<String>> {
    let references = find_cross_file_references(config, file_path)?;
    let mut related_files = std::collections::HashSet::new();

    for ref_info in references {
        related_files.insert(ref_info.from_file);
    }

    Ok(related_files.into_iter().collect())
}

pub fn preview_batch_operations(operations: &[BatchEditOperation]) -> BatchPreview {
    BatchPreview {
        operations: operations.to_vec(),
        total_files: operations.len(),
        estimated_changes: operations.len(),
    }
}

pub fn execute_batch_operations(
    operations: &[BatchEditOperation],
    workspace_root: &Path,
    dry_run: bool,
) -> Result<BatchResult> {
    let mut successful = Vec::new();
    let mut failed = Vec::new();
    let mut skipped = Vec::new();

    for op in operations {
        let full_path = workspace_root.join(&op.file_path);
        
        if !full_path.exists() {
            failed.push(FailedOperation {
                file_path: op.file_path.clone(),
                error: "File not found".to_string(),
            });
            continue;
        }

        if dry_run {
            skipped.push(op.file_path.clone());
            continue;
        }

        match apply_edit_operation(&full_path, &op.operation) {
            Ok(_) => successful.push(op.file_path.clone()),
            Err(e) => failed.push(FailedOperation {
                file_path: op.file_path.clone(),
                error: e.to_string(),
            }),
        }
    }

    Ok(BatchResult {
        successful,
        failed,
        skipped,
    })
}

fn apply_edit_operation(path: &Path, operation: &EditOperation) -> Result<()> {
    let content = fs::read_to_string(path)?;
    let new_content = match operation {
        EditOperation::Replace { old_string, new_string } => {
            if !content.contains(old_string) {
                return Err(anyhow!("Old string not found in file"));
            }
            content.replace(old_string, new_string)
        }
        EditOperation::Insert { position, content: insert_content } => {
            let lines: Vec<&str> = content.lines().collect();
            let mut new_lines = lines.clone();
            
            match position {
                InsertPosition::AfterLine { line } => {
                    if *line >= lines.len() {
                        return Err(anyhow!("Line number out of range"));
                    }
                    new_lines.insert(line + 1, insert_content);
                }
                InsertPosition::BeforeLine { line } => {
                    if *line >= lines.len() {
                        return Err(anyhow!("Line number out of range"));
                    }
                    new_lines.insert(*line, insert_content);
                }
                InsertPosition::AtEnd => {
                    new_lines.push(insert_content);
                }
                InsertPosition::AtStart => {
                    new_lines.insert(0, insert_content);
                }
            }
            new_lines.join("\n")
        }
        EditOperation::Delete { pattern } => {
            if !content.contains(pattern) {
                return Err(anyhow!("Pattern not found in file"));
            }
            content.replace(pattern, "")
        }
    };

    fs::write(path, new_content)?;
    Ok(())
}

pub fn create_backup(path: &Path) -> Result<PathBuf> {
    let backup_path = path.with_extension(format!("{}.bak", path.extension().unwrap_or_default().to_str().unwrap_or("")));
    fs::copy(path, &backup_path)?;
    Ok(backup_path)
}

pub fn restore_backup(backup_path: &Path, original_path: &Path) -> Result<()> {
    fs::copy(backup_path, original_path)?;
    fs::remove_file(backup_path)?;
    Ok(())
}