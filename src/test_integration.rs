use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TestDiscovery {
    pub framework: String,
    pub test_files: Vec<String>,
    pub run_command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TestResult {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub failures: Vec<String>,
    pub exit_code: i32,
    pub output: String,
}

pub fn discover_tests(workspace_root: &str) -> Result<TestDiscovery> {
    let root = Path::new(workspace_root);
    
    // Check for Python/pytest
    if root.join("pytest.ini").exists() 
        || root.join("pyproject.toml").exists()
        || root.join("setup.py").exists()
        || root.join("requirements.txt").exists() {
        
        let test_files = find_python_test_files(root)?;
        if !test_files.is_empty() {
            return Ok(TestDiscovery {
                framework: "pytest".to_string(),
                test_files,
                run_command: Some("pytest".to_string()),
            });
        }
    }
    
    // Check for Rust/cargo
    if root.join("Cargo.toml").exists() {
        let test_files = find_rust_test_files(root)?;
        if !test_files.is_empty() {
            return Ok(TestDiscovery {
                framework: "cargo".to_string(),
                test_files,
                run_command: Some("cargo test".to_string()),
            });
        }
    }
    
    // Check for JavaScript/npm
    if root.join("package.json").exists() {
        let test_files = find_js_test_files(root)?;
        if !test_files.is_empty() {
            let package_json = root.join("package.json");
            if let Ok(content) = fs::read_to_string(&package_json) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(scripts) = json.get("scripts").and_then(|s| s.as_object()) {
                        if scripts.contains_key("test") {
                            return Ok(TestDiscovery {
                                framework: "npm".to_string(),
                                test_files,
                                run_command: Some("npm test".to_string()),
                            });
                        }
                    }
                }
            }
        }
    }
    
    // Check for Go
    if root.join("go.mod").exists() {
        let test_files = find_go_test_files(root)?;
        if !test_files.is_empty() {
            return Ok(TestDiscovery {
                framework: "go".to_string(),
                test_files,
                run_command: Some("go test ./...".to_string()),
            });
        }
    }
    
    Ok(TestDiscovery {
        framework: "unknown".to_string(),
        test_files: vec![],
        run_command: None,
    })
}

fn find_python_test_files(root: &Path) -> Result<Vec<String>> {
    let mut test_files = Vec::new();
    
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "py") {
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if filename.starts_with("test_") || filename.ends_with("_test.py") {
                if let Ok(rel) = path.strip_prefix(root) {
                    test_files.push(rel.display().to_string());
                }
            }
        }
    }
    
    Ok(test_files)
}

fn find_rust_test_files(root: &Path) -> Result<Vec<String>> {
    let mut test_files = Vec::new();
    
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "rs") {
            if let Ok(content) = fs::read_to_string(path) {
                if content.contains("#[test]") || content.contains("#[cfg(test)]") {
                    if let Ok(rel) = path.strip_prefix(root) {
                        test_files.push(rel.display().to_string());
                    }
                }
            }
        }
    }
    
    Ok(test_files)
}

fn find_js_test_files(root: &Path) -> Result<Vec<String>> {
    let mut test_files = Vec::new();
    
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "js" || ext == "ts") {
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if filename.contains(".test.") || filename.contains(".spec.") {
                if let Ok(rel) = path.strip_prefix(root) {
                    test_files.push(rel.display().to_string());
                }
            }
        }
    }
    
    Ok(test_files)
}

fn find_go_test_files(root: &Path) -> Result<Vec<String>> {
    let mut test_files = Vec::new();
    
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "go") {
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if filename.ends_with("_test.go") {
                if let Ok(rel) = path.strip_prefix(root) {
                    test_files.push(rel.display().to_string());
                }
            }
        }
    }
    
    Ok(test_files)
}

pub fn run_tests(workspace_root: &str, pattern: Option<&str>) -> Result<TestResult> {
    let root = Path::new(workspace_root);
    let discovery = discover_tests(workspace_root)?;
    
    if discovery.framework == "unknown" {
        return Ok(TestResult {
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            failures: vec!["No test framework detected".to_string()],
            exit_code: 1,
            output: String::new(),
        });
    }
    
    let command = discovery.run_command.unwrap_or_default();
    let full_command = if let Some(pattern) = pattern {
        format!("{} -k {}", command, pattern)
    } else {
        command
    };
    
    let output = Command::new("sh")
        .arg("-c")
        .arg(&full_command)
        .current_dir(root)
        .output()?;
    
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined_output = format!("{}\n{}", stdout, stderr);
    
    let result = parse_test_results(&combined_output, &discovery.framework);
    
    Ok(TestResult {
        output: combined_output,
        exit_code: output.status.code().unwrap_or(1),
        ..result
    })
}

pub fn parse_test_results(output: &str, framework: &str) -> TestResult {
    match framework {
        "pytest" => parse_pytest_output(output),
        "cargo" => parse_cargo_output(output),
        "npm" => parse_npm_output(output),
        "go" => parse_go_output(output),
        _ => TestResult {
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            failures: vec![],
            exit_code: 1,
            output: String::new(),
        },
    }
}

fn parse_pytest_output(output: &str) -> TestResult {
    let mut result = TestResult::default();
    
    // Parse pytest summary
    for line in output.lines() {
        if line.contains(" passed") {
            if let Some(passed) = extract_number(line, "passed") {
                result.passed = passed;
            }
        }
        if line.contains(" failed") {
            if let Some(failed) = extract_number(line, "failed") {
                result.failed = failed;
            }
        }
        if line.contains(" skipped") {
            if let Some(skipped) = extract_number(line, "skipped") {
                result.skipped = skipped;
            }
        }
    }
    
    result.total = result.passed + result.failed + result.skipped;
    
    // Extract failure messages
    let mut failures = Vec::new();
    let mut in_failure = false;
    let mut current_failure = String::new();
    
    for line in output.lines() {
        if line.contains("FAILED") {
            in_failure = true;
            if !current_failure.is_empty() {
                failures.push(current_failure.clone());
                current_failure.clear();
            }
            current_failure.push_str(line);
        } else if in_failure {
            if line.is_empty() || line.starts_with("=") {
                in_failure = false;
                if !current_failure.is_empty() {
                    failures.push(current_failure.clone());
                    current_failure.clear();
                }
            } else {
                current_failure.push_str("\n");
                current_failure.push_str(line);
            }
        }
    }
    
    if !current_failure.is_empty() {
        failures.push(current_failure);
    }
    
    result.failures = failures;
    result
}

fn parse_cargo_output(output: &str) -> TestResult {
    let mut result = TestResult::default();
    
    for line in output.lines() {
        if line.contains("test result:") {
            if let Some(passed) = extract_number(line, "passed") {
                result.passed = passed;
            }
            if let Some(failed) = extract_number(line, "failed") {
                result.failed = failed;
            }
        }
    }
    
    result.total = result.passed + result.failed + result.skipped;
    
    // Extract failures
    let mut failures = Vec::new();
    let mut in_failure = false;
    let mut current_failure = String::new();
    
    for line in output.lines() {
        if line.contains("FAILED") {
            in_failure = true;
            if !current_failure.is_empty() {
                failures.push(current_failure.clone());
                current_failure.clear();
            }
            current_failure.push_str(line);
        } else if in_failure {
            if line.starts_with("test result:") || line.is_empty() {
                in_failure = false;
                if !current_failure.is_empty() {
                    failures.push(current_failure.clone());
                    current_failure.clear();
                }
            } else {
                current_failure.push_str("\n");
                current_failure.push_str(line);
            }
        }
    }
    
    if !current_failure.is_empty() {
        failures.push(current_failure);
    }
    
    result.failures = failures;
    result
}

fn parse_npm_output(output: &str) -> TestResult {
    let mut result = TestResult::default();
    
    // Try to extract from common test reporters
    for line in output.lines() {
        if line.contains("passing") || line.contains("✓") {
            result.passed += 1;
        }
        if line.contains("failing") || line.contains("✗") {
            result.failed += 1;
        }
    }
    
    result.total = result.passed + result.failed + result.skipped;
    result
}

fn parse_go_output(output: &str) -> TestResult {
    let mut result = TestResult::default();
    
    for line in output.lines() {
        if line.contains("PASS") {
            result.passed += 1;
        }
        if line.contains("FAIL") {
            result.failed += 1;
        }
    }
    
    result.total = result.passed + result.failed + result.skipped;
    result
}

fn extract_number(line: &str, keyword: &str) -> Option<usize> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == keyword || part.contains(keyword) {
            if i > 0 {
                if let Ok(num) = parts[i - 1].parse::<usize>() {
                    return Some(num);
                }
            }
            if i + 1 < parts.len() {
                if let Ok(num) = parts[i + 1].parse::<usize>() {
                    return Some(num);
                }
            }
        }
    }
    None
}

impl Default for TestResult {
    fn default() -> Self {
        Self {
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            failures: vec![],
            exit_code: 0,
            output: String::new(),
        }
    }
}

pub fn smart_test_selection(workspace_root: &str) -> Result<Vec<String>> {
    let root = Path::new(workspace_root);
    
    // Get git changes
    let output = Command::new("git")
        .args(["diff", "--name-only", "HEAD"])
        .current_dir(root)
        .output()?;
    
    if !output.status.success() {
        // If git fails, return all tests
        let discovery = discover_tests(workspace_root)?;
        return Ok(discovery.test_files);
    }
    
    let changed_files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.is_empty())
        .map(|s| s.to_string())
        .collect();
    
    if changed_files.is_empty() {
        return Ok(vec![]);
    }
    
    let discovery = discover_tests(workspace_root)?;
    let mut selected_tests = Vec::new();
    
    for changed_file in changed_files {
        for test_file in &discovery.test_files {
            // Simple heuristic: if test file is in same directory or has similar name
            let changed_base = changed_file.replace(".rs", "").replace(".py", "").replace(".js", "");
            let test_file_matches = test_file.contains(&changed_base);
            
            let dir_match = if let Some(first_part) = changed_file.split('/').next() {
                test_file.starts_with(first_part)
            } else {
                false
            };
            
            if test_file_matches || dir_match {
                if !selected_tests.contains(test_file) {
                    selected_tests.push(test_file.clone());
                }
            }
        }
    }
    
    Ok(selected_tests)
}