use anyhow::{anyhow, Result};
use clap::Parser;
use colored::*;
use html2text::from_read;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::timeout;
use tracing::error;
use tracing_subscriber::EnvFilter;
use url::Url;

use lazy_static::lazy_static;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;
use tokio::sync::RwLock;

/// Represents a single background process we spawned.
struct BackgroundProcess {
    command: String,
    label: String,
    stdout_lines: Vec<String>,
    stderr_lines: Vec<String>,
    status: BackgroundStatus,

    // The child handle (we keep so we can poll or kill it).
    child: Option<Child>,

    // A JoinHandle for reading stdout/stderr in the background
    io_task: Option<tokio::task::JoinHandle<()>>,
}

/// Process status: Running, Completed, or Error
#[derive(Debug, Clone)]
enum BackgroundStatus {
    Running,
    Completed(i32), // exit code
    Error(String),  // error message
}

impl BackgroundProcess {
    fn new(command: String, label: String, child: Child) -> Self {
        Self {
            command,
            label,
            stdout_lines: vec![],
            stderr_lines: vec![],
            status: BackgroundStatus::Running,
            child: Some(child),
            io_task: None,
        }
    }
}

/// A global manager that tracks background processes by label.
struct BackgroundProcessManager {
    processes: HashMap<String, BackgroundProcess>,
}

impl BackgroundProcessManager {
    fn new() -> Self {
        Self {
            processes: HashMap::new(),
        }
    }

    /// Spawn a new background process if the label is not already in use.
    /// Returns an error if the label is taken or process spawn fails.
    async fn spawn_process(&mut self, label: &str, command: &str) -> anyhow::Result<()> {
        if self.processes.contains_key(label) {
            return Err(anyhow::anyhow!(
                "Background label '{}' is already in use!",
                label
            ));
        }

        // For safety, check forbidden commands
        let forbidden_commands = ["rm", "mv", "dd", "shutdown", "reboot", "kill"];
        if forbidden_commands.iter().any(|&cmd| command.contains(cmd)) {
            return Err(anyhow::anyhow!(
                "Command contains forbidden operation: {}",
                command
            ));
        }

        // Spawn child
        let mut child = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Wrap child in our data structure
        let mut bp = BackgroundProcess::new(command.to_string(), label.to_string(), child);

        // Start an I/O monitoring task that reads stdout/stderr lines as they come
        let child_stdout = bp
            .child
            .as_mut()
            .unwrap()
            .stdout
            .take()
            .map(BufReader::new)
            .unwrap();
        let child_stderr = bp
            .child
            .as_mut()
            .unwrap()
            .stderr
            .take()
            .map(BufReader::new)
            .unwrap();

        // label for the async move
        let label_clone = label.to_string();

        let io_task = tokio::spawn({
            let mut stdout_reader = child_stdout.lines();
            let mut stderr_reader = child_stderr.lines();

            async move {
                loop {
                    tokio::select! {
                        // read next line from stdout
                        maybe_stdout_line = stdout_reader.next_line() => {
                            match maybe_stdout_line {
                                Ok(Some(line)) => {
                                    // Store the line in the manager
                                    let mut pm = PROCESS_MANAGER.write().await;
                                    if let Some(proc) = pm.processes.get_mut(&label_clone) {
                                        proc.stdout_lines.push(line);
                                    }
                                },
                                Ok(None) => {
                                    // stdout is EOF
                                },
                                Err(e) => {
                                    let mut pm = PROCESS_MANAGER.write().await;
                                    if let Some(proc) = pm.processes.get_mut(&label_clone) {
                                        proc.stderr_lines.push(format!("Error reading stdout: {}", e));
                                    }
                                }
                            }
                        },

                        // read next line from stderr
                        maybe_stderr_line = stderr_reader.next_line() => {
                            match maybe_stderr_line {
                                Ok(Some(line)) => {
                                    let mut pm = PROCESS_MANAGER.write().await;
                                    if let Some(proc) = pm.processes.get_mut(&label_clone) {
                                        proc.stderr_lines.push(line);
                                    }
                                },
                                Ok(None) => {
                                    // stderr is EOF
                                },
                                Err(e) => {
                                    let mut pm = PROCESS_MANAGER.write().await;
                                    if let Some(proc) = pm.processes.get_mut(&label_clone) {
                                        proc.stderr_lines.push(format!("Error reading stderr: {}", e));
                                    }
                                }
                            }
                        },

                        // If both streams have closed, break.
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        bp.io_task = Some(io_task);
        self.processes.insert(label.to_string(), bp);

        Ok(())
    }

    /// Check if a background process is still running, finished, or in error.
    /// Also updates the stored exit code if completed.
    async fn refresh_status(&mut self, label: &str) -> anyhow::Result<BackgroundStatus> {
        let maybe_proc = self.processes.get_mut(label);
        if maybe_proc.is_none() {
            return Err(anyhow::anyhow!(
                "No background process with label '{}'",
                label
            ));
        }
        let bp = maybe_proc.unwrap();

        if let BackgroundStatus::Running = bp.status {
            // If still "Running", check if the child is done
            if let Some(child) = bp.child.as_mut() {
                match child.try_wait()? {
                    Some(status) => {
                        // The process ended
                        let code = status.code().unwrap_or(-1);
                        bp.status = BackgroundStatus::Completed(code);
                        // We can close the child's handle to free resources
                        bp.child = None;
                    }
                    None => {
                        // Still running
                    }
                }
            }
        }
        Ok(bp.status.clone())
    }

    /// Return the logs and status for the given label
    async fn get_logs(
        &mut self,
        label: &str,
    ) -> anyhow::Result<(BackgroundStatus, String, String)> {
        self.refresh_status(label).await?; // update status
        let maybe_proc = self.processes.get(label);
        if maybe_proc.is_none() {
            return Err(anyhow::anyhow!(
                "No background process with label '{}'",
                label
            ));
        }
        let bp = maybe_proc.unwrap();
        let stdout = bp.stdout_lines.join("\n");
        let stderr = bp.stderr_lines.join("\n");
        Ok((bp.status.clone(), stdout, stderr))
    }

    /// Terminate a running process (if it is still running).
    /// This will mark status as Error if we forcibly kill it.
    async fn terminate(&mut self, label: &str) -> anyhow::Result<()> {
        let maybe_proc = self.processes.get_mut(label);
        if maybe_proc.is_none() {
            return Err(anyhow::anyhow!(
                "No background process with label '{}'",
                label
            ));
        }
        let bp = maybe_proc.unwrap();
        if let Some(child) = bp.child.as_mut() {
            // Attempt to kill it
            child.kill().await?;
            bp.status = BackgroundStatus::Error("Terminated by user".to_string());
            bp.child = None;
        }
        Ok(())
    }
}

lazy_static! {
    /// A global read/write lock to manage background processes
    static ref PROCESS_MANAGER: RwLock<BackgroundProcessManager> = RwLock::new(BackgroundProcessManager::new());
}

//
// -----------------------------------------------------------------------------
// HTML -> Text Extraction
// -----------------------------------------------------------------------------

pub fn extract_text_from_html(html: &str, url: Option<&str>) -> String {
    // Limit HTML size to prevent stack overflows
    const MAX_HTML_SIZE: usize = 3_000_000; // 100KB
    let html = if html.len() > MAX_HTML_SIZE {
        &html[..MAX_HTML_SIZE]
    } else {
        html
    };

    // Use a Box to move the HTML processing to the heap
    let text = Box::new(move || {
        from_read(html.as_bytes(), 80).unwrap_or_else(|_| "Failed to parse HTML".into())
    })();

    // If URL provided, add it as reference
    if let Some(url_str) = url {
        if let Ok(parsed_url) = Url::parse(url_str) {
            let mut text = text;
            text.push_str(&format!("\n\nSource: {}", url_str));

            if let Some(domain) = parsed_url.domain() {
                text.push_str(&format!("\nDomain: {}", domain));
            }
            return text;
        }
    }

    text
}

//
// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

/// Represents a single message in the conversation: system, user, or assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", content = "content")]
pub enum Message {
    System(String),
    User(String),
    Assistant(String),
}

/// We allow up to this many consecutive "steps" of calling the LLM automatically
/// when running multi-step tasks (like chain-of-thought).
const MAX_ITERATIONS: usize = 10;

/// Maximum approximate character count before we trigger conversation summarization.
/// (64k tokens ~ ~240k characters, so we'll use 200k as a rough threshold just to be safe).
const CONVERSATION_CHAR_LIMIT: usize = 200_000;

/// Maximum size for tool results to be stored in session (100KB)
const MAX_TOOL_RESULT_SIZE: usize = 100_000;

/// The entire conversation session, which we store in JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSession {
    pub messages: Vec<Message>,

    /// A set of small key-value variables (name -> (value, timestamp))
    pub variables: HashMap<String, (String, chrono::DateTime<chrono::Utc>)>,

    /// A flexible "long context" knowledge base, stored as JSON.
    /// Tools and the AI can append summarized data here (e.g. discovered hosts, vulnerabilities).
    pub long_context: Value,
}

impl Default for ConversationSession {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationSession {
    pub fn new() -> Self {
        Self {
            messages: vec![],
            variables: HashMap::new(),
            long_context: Value::Object(serde_json::Map::new()),
        }
    }

    /// Get a variable value by key
    pub fn get_var(&self, key: &str) -> Option<&String> {
        self.variables.get(key).map(|(value, _)| value)
    }

    /// Set a variable value with timestamp
    pub fn set_var(&mut self, key: String, value: String) {
        let now = chrono::Utc::now();
        self.variables.insert(key, (value, now));
    }

    /// List all variables with timestamps
    pub fn list_vars(&self) -> String {
        if self.variables.is_empty() {
            return "No variables stored".to_string();
        }

        // Sort by timestamp (newest first)
        let mut vars: Vec<_> = self.variables.iter().map(|(k, (v, t))| (t, k, v)).collect();
        vars.sort_by(|a, b| b.0.cmp(a.0));

        vars.iter()
            .map(|(t, k, v)| format!("{}: {} (set at {})", k, v, t.format("%Y-%m-%d %H:%M:%S")))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get the 10 most recent variables for context
    pub fn recent_vars(&self) -> String {
        if self.variables.is_empty() {
            return String::new();
        }

        // Sort by timestamp (newest first)
        let mut vars: Vec<_> = self.variables.iter().map(|(k, (v, t))| (t, k, v)).collect();
        vars.sort_by(|a, b| b.0.cmp(a.0));

        // Take top 10
        vars.iter()
            .take(10)
            .map(|(t, k, v)| format!("{}: {} (set at {})", k, v, t.format("%Y-%m-%d %H:%M:%S")))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Append data to our "long_context" knowledge base. For example:
    /// session.append_to_long_context("tool_summaries", &json!({ "timestamp": "now", "summary": "some summary" }));
    pub fn append_to_long_context(&mut self, key: &str, value: &Value) {
        // Ensure long_context is an object
        let obj = self.long_context.as_object_mut().unwrap();

        // If the key doesn't exist, create an empty array
        let entry = obj.entry(key.to_string()).or_insert(Value::Array(vec![]));

        // Append to array
        if let Value::Array(arr) = entry {
            arr.push(value.clone());
        }
    }

    /// Return a summarized view of the long_context. For demonstration,
    /// we'll just JSON-stringify it. In a real system, you might keep this more condensed.
    pub fn summarize_long_context(&self) -> String {
        let s = serde_json::to_string_pretty(&self.long_context).unwrap_or_else(|_| "".to_string());
        if s.trim().is_empty() || s == "{}" {
            "".to_string()
        } else {
            format!("Knowledge Base:\n{}", s)
        }
    }
}
#[derive(Debug)]
pub enum ToolCall {
    Search {
        query: String,
        count: Option<u8>,
        safe_search: Option<bool>,
    },
    Bash {
        command: String,
        args: Vec<String>,
    },
    /// <bash command="..." background="true" label="myjob" />
    BashBackground {
        label: String,
        command: String,
        args: Vec<String>,
    },

    /// Check a background job’s status: <bash_status label="myjob" />
    BashStatus {
        label: String,
    },

    /// Get the logs for a background job: <bash_logs label="myjob" />
    BashLogs {
        label: String,
    },

    /// Terminate a background job: <bash_terminate label="myjob" />
    BashTerminate {
        label: String,
    },

    Scrape {
        url: String,
        render_js: Option<bool>,
    },
    GetVar {
        name: String,
    },
    SetVar {
        name: String,
        value: String,
    },
    ListVars,
}

//
// -----------------------------------------------------------------------------
// Reflection Blocks
// -----------------------------------------------------------------------------

fn extract_reflection_blocks(text: &str) -> Vec<String> {
    let re_reflection = Regex::new(r"(?s)<reflection>(.*?)</reflection>").unwrap();
    let mut blocks = Vec::new();
    for cap in re_reflection.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            blocks.push(m.as_str().trim().to_string());
        }
    }
    blocks
}

//
// -----------------------------------------------------------------------------
// Summaries of Repetitive Tool Output
// -----------------------------------------------------------------------------

/// This function calls the LLM in a minimal "summarizer" context, returning bullet points.
async fn summarize_tool_output(text: &str) -> Result<String> {
    // If it's empty or short, just skip summarizing
    if text.trim().is_empty() {
        return Ok("".to_string());
    }

    // Build a minimal session to ask for a summary
    let mut summary_sess = ConversationSession::new();
    summary_sess.messages.push(Message::System(
        "You are a summarizer. Summarize the following tool output in bullet points, focusing on key/unique items and brevity.".to_string()
    ));
    summary_sess.messages.push(Message::User(text.to_string()));

    // We reuse the normal deepseek-chat for summarizing tool outputs
    let summary = call_deepseek_api(&mut summary_sess, "deepseek-chat").await?;
    Ok(summary.trim().to_string())
}

//
// -----------------------------------------------------------------------------
// Minimal Tool Implementations (Search, Bash, Scrape)
// -----------------------------------------------------------------------------

// -- Bash Tool
async fn run_bash_command(command: &str) -> Result<String> {
    // Validate command for safety
    let forbidden_commands = ["rm", "mv", "dd", "shutdown", "reboot", "kill"];
    if forbidden_commands.iter().any(|&cmd| command.contains(cmd)) {
        return Err(anyhow!("Command contains forbidden operation: {}", command));
    }

    // Run command with timeout
    let output = tokio::time::timeout(
        Duration::from_secs(30),
        tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .output(),
    )
    .await??;

    // Format output more cleanly
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let code = output.status.code().unwrap_or(-1);

    let mut result = String::new();
    result.push_str(&format!("Command: {}\n", command.green().bold()));
    result.push_str(&format!("Exit Code: {}\n", code));

    if !stdout.is_empty() {
        result.push_str(&format!("\n{}:\n{}\n", "STDOUT".blue().bold(), stdout));
    }
    if !stderr.is_empty() {
        result.push_str(&format!("\n{}:\n{}\n", "STDERR".red().bold(), stderr));
    }

    Ok(result)
}

// -- Search Tool
async fn run_search(query: &str, count: Option<u8>, safe_search: Option<bool>) -> Result<String> {
    // For demonstration, do a simple call to Brave Search if an API key is set:
    let api_key = env::var("BRAVE_SEARCH_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        return Ok(
            "No BRAVE_SEARCH_API_KEY found. Provide one to enable real searching.".to_string(),
        );
    }

    let client = reqwest::Client::new();
    let url = "https://api.search.brave.com/res/v1/web/search";

    let count = count.unwrap_or(10);
    let safe = safe_search.unwrap_or(true);
    let safesearch = if safe { "strict" } else { "off" };
    let params = [
        ("q", query),
        ("count", &count.to_string()),
        ("safesearch", safesearch),
    ];

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("X-Subscription-Token", api_key.parse()?);

    let resp = client
        .get(url)
        .headers(headers)
        .query(&params)
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await?;
        return Err(anyhow!("Brave Search error: {text}"));
    }

    let json_response: Value = resp.json().await?;
    let mut result = String::new();
    result.push_str(&format!("Search Query: {}\n\n", query));

    if let Some(web_results) = json_response.get("web").and_then(|w| w.get("results")) {
        if let Some(results) = web_results.as_array() {
            for (i, item) in results.iter().enumerate() {
                if i >= 5 {
                    break;
                }
                if let (Some(url_v), Some(desc_v)) = (item.get("url"), item.get("description")) {
                    if let (Some(url_str), Some(desc_str)) = (url_v.as_str(), desc_v.as_str()) {
                        result.push_str(&format!(
                            "• {}: {}\n\n",
                            url_str.blue().underline(),
                            desc_str
                        ));
                    }
                }
            }
        }
    }

    Ok(result)
}

// -- Scrape Tool
async fn run_scrape(url: &str, render_js: bool) -> Result<String> {
    tracing::debug!("Starting scrape of URL: {}", url);

    // For real usage, put your ScrapingBee API key in SCRAPINGBEE_API_KEY
    let api_key = env::var("SCRAPINGBEE_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        tracing::warn!("No SCRAPINGBEE_API_KEY found - scraping disabled");
        return Ok(
            "No SCRAPINGBEE_API_KEY found. Provide one to enable real scraping.".to_string(),
        );
    }

    tracing::debug!("Initializing HTTP client");
    let client = reqwest::Client::new();
    let base_url = "https://app.scrapingbee.com/api/v1/";

    tracing::debug!("Building request with params: render_js={}", render_js);
    let request = client.get(base_url).query(&[
        ("api_key", &api_key),
        ("url", &url.to_string()),
        ("render_js", &render_js.to_string()),
    ]);

    tracing::debug!("Sending request to ScrapingBee API");
    let resp = match request.send().await {
        Ok(r) => {
            tracing::debug!("Received response with status: {}", r.status());
            r
        }
        Err(e) => {
            tracing::error!("Request failed: {}", e);
            return Err(anyhow!("Request failed: {}", e));
        }
    };

    if !resp.status().is_success() {
        tracing::warn!("Non-success status: {}", resp.status());
        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("Failed to read error response text: {}", e);
                return Err(anyhow!("ScrapingBee error: {}", e));
            }
        };
        tracing::debug!("Error response text: {}", text);
        return Err(anyhow!("ScrapingBee error: {text}"));
    }

    tracing::debug!("Checking content type");
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    tracing::debug!("Content type: {}", content_type);

    if content_type.contains("text") || content_type.contains("json") {
        tracing::debug!("Processing text response");
        let text = match resp.text().await {
            Ok(t) => {
                tracing::debug!("Received text response of length: {}", t.len());
                t
            }
            Err(e) => {
                tracing::error!("Failed to read response text: {}", e);
                return Err(anyhow!("Failed to read response: {}", e));
            }
        };

        tracing::debug!("Extracting text from HTML");
        let readable_text = extract_text_from_html(&text, Some(url));
        tracing::debug!("Extracted text length: {}", readable_text.len());

        let result = format!("Scraped URL: {}\nExtracted Text:\n{}", url, readable_text);
        Ok(result)
    } else {
        tracing::debug!("Processing binary response");
        let bytes = match resp.bytes().await {
            Ok(b) => {
                tracing::debug!("Received binary response of length: {}", b.len());
                b
            }
            Err(e) => {
                tracing::error!("Failed to read response bytes: {}", e);
                return Err(anyhow!("Failed to read binary response: {}", e));
            }
        };

        let result = format!(
            "Scraped URL: {}\nReceived {} bytes of binary data.",
            url,
            bytes.len()
        );
        Ok(result)
    }
}

//
// -----------------------------------------------------------------------------
// Parsing the Assistant's Tool Calls (<search>, <bash>, <scrape>, etc.)
// -----------------------------------------------------------------------------

fn parse_tool_calls(assistant_text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Example usage:
    // <search query="stuff" count="3" safe_search="false" />
    let re_search = Regex::new(
        r#"(?s)<search\s+query="([^"]+)"(?:\s+count="(\d+)")?(?:\s+safe_search="(true|false)")?\s*/>"#,
    )
    .unwrap();
    for cap in re_search.captures_iter(assistant_text) {
        let query = cap[1].trim().to_string();
        let count = cap.get(2).and_then(|m| m.as_str().parse::<u8>().ok());
        let safe_search = cap.get(3).and_then(|m| m.as_str().parse::<bool>().ok());
        calls.push(ToolCall::Search {
            query,
            count,
            safe_search,
        });
    }

    // Enhanced BASH with optional background and label
    // e.g. <bash command="ls" args="-l /tmp" background="true" label="job1" />
    let re_bash_background = Regex::new(
        r#"(?s)<bash\s+command="([^"]+)"(?:\s+args="([^"]*)")?(?:\s+background="(true|false)")?(?:\s+label="([^"]+)")?\s*/>"#,
    )
    .unwrap();
    for cap in re_bash_background.captures_iter(assistant_text) {
        let command = cap[1].trim().to_string();
        let args = cap
            .get(2)
            .map(|m| {
                m.as_str()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_else(|| Vec::new());
        let background = cap.get(3).map(|m| m.as_str() == "true").unwrap_or(false);
        let label = cap
            .get(4)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();

        if background {
            // It's a background bash call
            let label_used = if label.is_empty() {
                // If user forgot label, generate one or raise an error
                format!("job_{}", uuid::Uuid::new_v4())
            } else {
                label
            };
            calls.push(ToolCall::BashBackground {
                label: label_used,
                command,
                args,
            });
        } else {
            // normal synchronous bash call
            calls.push(ToolCall::Bash { command, args });
        }
    }

    // <bash_status label="myjob" />
    let re_bash_status = Regex::new(r#"(?s)<bash_status\s+label="([^"]+)"\s*/>"#).unwrap();
    for cap in re_bash_status.captures_iter(assistant_text) {
        let label = cap[1].trim().to_string();
        calls.push(ToolCall::BashStatus { label });
    }

    // <bash_logs label="myjob" />
    let re_bash_logs = Regex::new(r#"(?s)<bash_logs\s+label="([^"]+)"\s*/>"#).unwrap();
    for cap in re_bash_logs.captures_iter(assistant_text) {
        let label = cap[1].trim().to_string();
        calls.push(ToolCall::BashLogs { label });
    }

    // <bash_terminate label="myjob" />
    let re_bash_terminate = Regex::new(r#"(?s)<bash_terminate\s+label="([^"]+)"\s*/>"#).unwrap();
    for cap in re_bash_terminate.captures_iter(assistant_text) {
        let label = cap[1].trim().to_string();
        calls.push(ToolCall::BashTerminate { label });
    }
    // Example usage:
    // <scrape url="https://example.com" render_js="true" />
    let re_scrape =
        Regex::new(r#"(?s)<scrape\s+url="([^"]+)"(?:\s+render_js="(true|false)")?\s*/>"#).unwrap();
    for cap in re_scrape.captures_iter(assistant_text) {
        let url = cap[1].trim().to_string();
        let render_js = cap.get(2).and_then(|m| m.as_str().parse::<bool>().ok());
        calls.push(ToolCall::Scrape { url, render_js });
    }

    // Example usage:
    // <get_var name="foo" />
    let re_get_var = Regex::new(r#"(?s)<get_var\s+name="([^"]+)"\s*/>"#).unwrap();
    for cap in re_get_var.captures_iter(assistant_text) {
        let name = cap[1].trim().to_string();
        calls.push(ToolCall::GetVar { name });
    }

    // Example usage:
    // <set_var name="foo" value="bar" />
    let re_set_var = Regex::new(r#"(?s)<set_var\s+name="([^"]+)"\s+value="([^"]+)"\s*/>"#).unwrap();
    for cap in re_set_var.captures_iter(assistant_text) {
        let name = cap[1].trim().to_string();
        let value = cap[2].trim().to_string();
        calls.push(ToolCall::SetVar { name, value });
    }

    // Example usage:
    // <list_vars />
    let re_list_vars = Regex::new(r"(?s)<list_vars\s*/>").unwrap();
    if re_list_vars.is_match(assistant_text) {
        calls.push(ToolCall::ListVars);
    }

    // Reflection blocks are extracted separately (not a ToolCall).
    calls
}

//
// -----------------------------------------------------------------------------
// LLM Calls
// -----------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct DeepSeekChatRequest {
    model: String,
    messages: Vec<DeepSeekMessage>,
    stream: bool,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct DeepSeekMessage {
    role: String,
    content: String,
}

/// Use "deepseek-chat" or "deepseek-reasoner" model as requested
/// (based on the `model_name` parameter).
///
/// Summarize the conversation if it exceeds a certain size (CONVERSATION_CHAR_LIMIT).
/// We'll do that by calling `summarize_conversation` with the "deepseek-reasoner" model
/// and replacing older messages with a summary message (keeping the original system message).
async fn call_deepseek_api(session: &mut ConversationSession, model_name: &str) -> Result<String> {
    // Box the future to handle recursion
    Box::pin(async move {
        // If conversation is too large, summarize and reduce it
        if estimate_conversation_size(&session) > CONVERSATION_CHAR_LIMIT {
            // Summarize the conversation so far
            let summary = summarize_conversation(session).await?;
            // Replace older messages with the system message + summary
            let original_system_msg = session
                .messages
                .iter()
                .find(|m| matches!(m, Message::System(_)))
                .cloned()
                .unwrap_or(Message::System(
                    "You are an EXPERT AI ASSISTANT".to_string(),
                ));

            // We'll keep only the system message + a new "assistant" summary +
            // the last user message if there's one (so we keep the context of the user's question).
            let last_user_message = session
                .messages
                .iter()
                .rev()
                .find(|m| matches!(m, Message::User(_)))
                .cloned();

            session.messages.clear();
            // Keep the original system message
            session.messages.push(original_system_msg);

            // Keep the summary as an "assistant" message
            session.messages.push(Message::Assistant(format!(
                "(Conversation Summary)\n{}",
                summary
            )));

            // Keep the last user message (if any) so the conversation makes sense
            if let Some(user_msg) = last_user_message {
                session.messages.push(user_msg);
            }
        }

        // Convert session messages -> DeepSeekMessage
        let mut ds_messages = Vec::new();
        for msg in &session.messages {
            match msg {
                Message::System(txt) => ds_messages.push(DeepSeekMessage {
                    role: "system".into(),
                    content: txt.clone(),
                }),
                Message::User(txt) => ds_messages.push(DeepSeekMessage {
                    role: "user".into(),
                    content: txt.clone(),
                }),
                Message::Assistant(txt) => ds_messages.push(DeepSeekMessage {
                    role: "assistant".into(),
                    content: txt.clone(),
                }),
            }
        }

        let req_body = DeepSeekChatRequest {
            model: model_name.to_string(),
            messages: ds_messages,
            stream: false,
            temperature: 0.7,
        };

        let api_key = env::var("DEEPSEEK_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            error!("Missing DEEPSEEK_API_KEY environment variable!");
            return Err(anyhow!("DEEPSEEK_API_KEY not set"));
        }

        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.deepseek.com/chat/completions")
            .bearer_auth(api_key)
            .json(&req_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let text = resp.text().await?;
            return Err(anyhow!("DeepSeek API Error: {}", text));
        }

        let json_resp: Value = resp.json().await?;
        // Usually: { "choices": [ { "message": {"role":"assistant","content":"hello"} } ], ... }
        if let Some(choices) = json_resp.get("choices") {
            if let Some(first_choice) = choices.get(0) {
                if let Some(msg) = first_choice.get("message") {
                    if let Some(content) = msg.get("content") {
                        let content_str = content.as_str().unwrap_or("").to_string();
                        return Ok(content_str);
                    }
                }
            }
        }

        Err(anyhow!("No valid content in DeepSeek API response"))
    })
    .await
}

//
// -----------------------------------------------------------------------------
// Summarize the Whole Conversation (using deepseek-reasoner)
// -----------------------------------------------------------------------------

/// Summarize the entire conversation while specifically excluding certain information
async fn forget_conversation(session: &ConversationSession, to_forget: &str) -> Result<String> {
    // Convert to text
    let mut conversation_text = String::new();
    for msg in &session.messages {
        match msg {
            Message::System(txt) => {
                conversation_text.push_str("[System]\n");
                conversation_text.push_str(txt);
                conversation_text.push_str("\n\n");
            }
            Message::User(txt) => {
                conversation_text.push_str("[User]\n");
                conversation_text.push_str(txt);
                conversation_text.push_str("\n\n");
            }
            Message::Assistant(txt) => {
                conversation_text.push_str("[Assistant]\n");
                conversation_text.push_str(txt);
                conversation_text.push_str("\n\n");
            }
        }
    }

    // Build a minimal session specifically for forgetting
    let mut forget_sess = ConversationSession::new();
    forget_sess.messages.push(Message::System(
        format!("You are a specialized summarizer (deepseek-reasoner). Summarize the following conversation while specifically EXCLUDING any information about '{}'. Focus on user goals/intent and key points, but ensure you do not mention or include anything related to '{}'. Be concise but do not lose critical context. Omit extraneous details.", to_forget, to_forget)
    ));

    // Then put the entire conversation in as user text:
    forget_sess.messages.push(Message::User(conversation_text));

    // Call the reasoner
    let summary = call_deepseek_api(&mut forget_sess, "deepseek-reasoner").await?;
    Ok(summary.trim().to_string())
}

/// Summarize the entire conversation, focusing on user intent and relevant findings.
/// This is called when the conversation gets too large.
async fn summarize_conversation(session: &ConversationSession) -> Result<String> {
    // Convert to text. We'll do something simplistic: user/assistant role labels plus message content.
    let mut conversation_text = String::new();
    for msg in &session.messages {
        match msg {
            Message::System(txt) => {
                conversation_text.push_str("[System]\n");
                conversation_text.push_str(txt);
                conversation_text.push_str("\n\n");
            }
            Message::User(txt) => {
                conversation_text.push_str("[User]\n");
                conversation_text.push_str(txt);
                conversation_text.push_str("\n\n");
            }
            Message::Assistant(txt) => {
                conversation_text.push_str("[Assistant]\n");
                conversation_text.push_str(txt);
                conversation_text.push_str("\n\n");
            }
        }
    }

    // We'll build a minimal session specifically for summarization, using the "deepseek-reasoner" model
    let mut summary_sess = ConversationSession::new();

    // Keep the same system message we've had since the beginning (or a simpler directive).
    // The user explicitly said "We always want to keep the system message the same" in normal usage,
    // but for summarizing the conversation, let's give the reasoner a direct instruction:
    summary_sess.messages.push(Message::System(
        "You are a specialized summarizer (deepseek-reasoner). Summarize the following conversation. 
Focus on user goals/intent, key points discovered, important data. Be concise but do not lose critical context. 
Omit extraneous details.".to_string()
    ));

    // Then put the entire conversation in as user text:
    summary_sess.messages.push(Message::User(conversation_text));

    // Call the reasoner
    let summary = call_deepseek_api(&mut summary_sess, "deepseek-reasoner").await?;
    Ok(summary.trim().to_string())
}

//
// -----------------------------------------------------------------------------
// Utility to estimate conversation size
// -----------------------------------------------------------------------------

fn estimate_conversation_size(session: &ConversationSession) -> usize {
    // Rough measure: total character length of all messages
    // Not a perfect token count, but good enough for a threshold check
    let mut total = 0;
    for msg in &session.messages {
        match msg {
            Message::System(txt) => total += txt.len(),
            Message::User(txt) => total += txt.len(),
            Message::Assistant(txt) => total += txt.len(),
        }
    }
    total
}

//
// -----------------------------------------------------------------------------
// Multi-Step Tool Calls with Reflection
// -----------------------------------------------------------------------------

/// Process tool calls in a loop, automatically re-invoking the LLM until <done/> or
/// until we hit the max iteration limit. This is especially useful for multi-step pen-testing tasks.
/// We always call `call_deepseek_api` with model="deepseek-chat" for normal conversation steps.
pub async fn process_tool_calls(
    session: &mut ConversationSession,
    initial_response: String,
) -> Result<String> {
    let mut current_response = initial_response;
    let mut iteration = 0;

    loop {
        iteration += 1;
        if iteration > MAX_ITERATIONS {
            return Ok(format!(
                "{}: Reached maximum number of iterations ({})",
                "STOPPED".red().bold(),
                MAX_ITERATIONS
            ));
        }

        // Check for completion condition
        if current_response.contains("<done/>") {
            return Ok(current_response.replace("<done/>", "").trim().to_string());
        }

        // Parse reflection blocks. We'll store them in the conversation (as "assistant" messages).
        let reflection_blocks = extract_reflection_blocks(&current_response);
        for block in reflection_blocks {
            session
                .messages
                .push(Message::Assistant(format!("(reflection)\n{}", block)));
        }

        // Parse and execute tool calls
        let tool_calls = parse_tool_calls(&current_response);
        if tool_calls.is_empty() {
            // No further tool calls to make. We'll consider this the final answer for now.
            return Ok(current_response.trim().to_string());
        }

        let mut tool_results_text = String::new();

        for call in tool_calls {
            match call {
                ToolCall::Bash { command, args } => {
                    let full_cmd = format!("{} {}", command, args.join(" "));
                    let result = match run_bash_command(&full_cmd).await {
                        Ok(r) => r,
                        Err(e) => format!("{}: {}", "BASH ERROR".red().bold(), e),
                    };
                    // Then show tool results
                    println!(
                        "{} {}",
                        "[Tool Results]".yellow().bold(),
                        format!(
                            "{}\n{}\n",
                            "==== [BASH COMMAND RESULTS] ====".yellow().bold(),
                            result
                        )
                    );
                    tool_results_text.push_str(&format!(
                        "{}\n{}\n",
                        "==== [BASH COMMAND RESULTS] ====".yellow().bold(),
                        result
                    ));
                }
                ToolCall::Search {
                    query,
                    count,
                    safe_search,
                } => {
                    let result = match run_search(&query, count, safe_search).await {
                        Ok(r) => r,
                        Err(e) => format!("ERROR during search: {}", e),
                    };
                    // Then show tool results
                    println!(
                        "{} {}",
                        "[Tool Results]".yellow().bold(),
                        format!(
                            "==== [SEARCH result for \"{}\"] ====\n{}\n\n",
                            query, result
                        )
                    );
                    tool_results_text.push_str(&format!(
                        "==== [SEARCH result for \"{}\"] ====\n{}\n\n",
                        query, result
                    ));
                }

                ToolCall::BashBackground {
                    label,
                    command,
                    args,
                } => {
                    let full_cmd = format!("{} {}", command, args.join(" "));
                    let result = match {
                        let mut pm = PROCESS_MANAGER.write().await;
                        pm.spawn_process(&label, &full_cmd).await
                    } {
                        Ok(_) => format!(
                            "Spawned background job '{}' with command: {}",
                            label, full_cmd
                        ),
                        Err(e) => format!("ERROR spawning background job '{}': {}", label, e),
                    };
                    tool_results_text
                        .push_str(&format!("==== [BASH BACKGROUND] ====\n{}\n\n", result));
                }

                ToolCall::BashStatus { label } => {
                    let status = {
                        let mut pm = PROCESS_MANAGER.write().await;
                        pm.refresh_status(&label).await
                    };
                    let result = match status {
                        Ok(st) => format!("Status for '{}': {:?}", label, st),
                        Err(e) => format!("ERROR checking status: {}", e),
                    };
                    tool_results_text.push_str(&format!("==== [BASH STATUS] ====\n{}\n\n", result));
                }

                ToolCall::BashLogs { label } => {
                    let logs = {
                        let mut pm = PROCESS_MANAGER.write().await;
                        pm.get_logs(&label).await
                    };
                    let result = match logs {
                        Ok((st, out, err)) => {
                            format!(
                                "Logs for '{}':\nStatus: {:?}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}\n",
                                label, st, out, err
                            )
                        }
                        Err(e) => format!("ERROR fetching logs: {}", e),
                    };
                    tool_results_text.push_str(&format!("==== [BASH LOGS] ====\n{}\n\n", result));
                }

                ToolCall::BashTerminate { label } => {
                    let result = {
                        let mut pm = PROCESS_MANAGER.write().await;
                        pm.terminate(&label).await
                    };
                    let msg = match result {
                        Ok(_) => format!("Terminated background job '{}'.", label),
                        Err(e) => format!("ERROR terminating '{}': {}", label, e),
                    };
                    tool_results_text.push_str(&format!("==== [BASH TERMINATE] ====\n{}\n\n", msg));
                }
                ToolCall::Scrape { url, render_js } => {
                    let result = match run_scrape(&url, render_js.unwrap_or(true)).await {
                        Ok(r) => r,
                        Err(e) => format!("ERROR during scrape: {}", e),
                    };
                    // Then show tool results
                    println!(
                        "{} {}",
                        "[Tool Results]".yellow().bold(),
                        format!("==== [SCRAPE result for \"{}\"] ====\n{}\n\n", url, result)
                    );
                    tool_results_text.push_str(&format!(
                        "==== [SCRAPE result for \"{}\"] ====\n{}\n\n",
                        url, result
                    ));
                }
                ToolCall::GetVar { name } => {
                    let result = match session.get_var(&name) {
                        Some(value) => format!("Variable '{}': {}", name, value),
                        None => format!("Variable '{}' not found", name),
                    };
                    tool_results_text
                        .push_str(&format!("==== [GET VAR result] ====\n{}\n\n", result));
                }
                ToolCall::SetVar { name, value } => {
                    session.set_var(name.clone(), value.clone());
                    tool_results_text.push_str(&format!(
                        "==== [SET VAR result] ====\nSet '{}' to '{}'\n\n",
                        name, value
                    ));
                }
                ToolCall::ListVars => {
                    let result = session.list_vars();
                    tool_results_text
                        .push_str(&format!("==== [LIST VARS result] ====\n{}\n\n", result));
                }
            }
        }

        // Summarize the tool results if they're large
        if tool_results_text.len() > 500 {
            if let Ok(summary) = summarize_tool_output(&tool_results_text).await {
                if !summary.is_empty() {
                    // Add to conversation
                    session.messages.push(Message::Assistant(format!(
                        "Summary of tool results:\n{}",
                        summary
                    )));
                    // Also store in long_context
                    session.append_to_long_context(
                        "tool_summaries",
                        &json!({
                            "timestamp": chrono::Utc::now().to_rfc3339(),
                            "summary": summary
                        }),
                    );
                }
            }
        }

        // Add tool results as an assistant message so the LLM can see it
        session.messages.push(Message::Assistant(format!(
            "Tool results:\n{}",
            tool_results_text
        )));

        // Now re-invoke the LLM with the updated conversation, using "deepseek-chat"
        let next_response = timeout(
            Duration::from_secs(300),
            call_deepseek_api(session, "deepseek-chat"),
        )
        .await??;

        current_response = next_response;
    }
}

//
// -----------------------------------------------------------------------------
// Session Persistence
// -----------------------------------------------------------------------------

/// Save the conversation to `session.json`.
fn save_session(session: &ConversationSession) -> Result<()> {
    let path = PathBuf::from("session.json");
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, session)?;
    Ok(())
}

/// Load an existing session from `session.json`, if it exists.
fn load_session() -> ConversationSession {
    let path = PathBuf::from("session.json");
    load_session_from_path(&path)
}

fn load_session_from_path(path: &PathBuf) -> ConversationSession {
    if path.exists() {
        if let Ok(file) = File::open(path) {
            if let Ok(sess) = serde_json::from_reader(file) {
                return sess;
            }
        }
    }
    ConversationSession::new()
}

//
// -----------------------------------------------------------------------------
// Test Mode - Run a session file and show results
// -----------------------------------------------------------------------------

/// Run a session file in test mode - process all messages and show results
/// (one-shot, not interactive).
async fn run_test_session(session_path: &str) -> Result<()> {
    let path = PathBuf::from(session_path);
    let mut session = load_session_from_path(&path);

    let messages = session.messages.clone();

    for msg in messages {
        if let Message::User(user_input) = msg {
            println!("{} {}", "[User]".green().bold(), user_input);

            let result = timeout(
                Duration::from_secs(300),
                call_deepseek_api(&mut session, "deepseek-chat"),
            )
            .await;
            let assistant_response = match result {
                Ok(Ok(content)) => content,
                Ok(Err(e)) => {
                    eprintln!("{}: {}", "[Error from LLM]".red(), e.to_string());
                    continue;
                }
                Err(_) => {
                    eprintln!("{}", "[Timed out waiting for LLM]".red());
                    continue;
                }
            };

            // Parse reflection blocks
            let reflection_blocks = extract_reflection_blocks(&assistant_response);
            for block in reflection_blocks {
                session
                    .messages
                    .push(Message::Assistant(format!("(reflection)\n{}", block)));
            }

            // Parse tool calls
            let tool_calls = parse_tool_calls(&assistant_response);
            let mut tool_results_text = String::new();
            for call in tool_calls {
                match call {
                    ToolCall::BashBackground {
                        label,
                        command,
                        args,
                    } => {
                        let full_cmd = format!("{} {}", command, args.join(" "));
                        let result = match {
                            let mut pm = PROCESS_MANAGER.write().await;
                            pm.spawn_process(&label, &full_cmd).await
                        } {
                            Ok(_) => format!(
                                "Spawned background job '{}' with command: {}",
                                label, full_cmd
                            ),
                            Err(e) => format!("ERROR spawning background job '{}': {}", label, e),
                        };
                        tool_results_text
                            .push_str(&format!("==== [BASH BACKGROUND] ====\n{}\n\n", result));
                    }

                    ToolCall::BashStatus { label } => {
                        let status = {
                            let mut pm = PROCESS_MANAGER.write().await;
                            pm.refresh_status(&label).await
                        };
                        let result = match status {
                            Ok(st) => format!("Status for '{}': {:?}", label, st),
                            Err(e) => format!("ERROR checking status: {}", e),
                        };
                        tool_results_text
                            .push_str(&format!("==== [BASH STATUS] ====\n{}\n\n", result));
                    }

                    ToolCall::BashLogs { label } => {
                        let logs = {
                            let mut pm = PROCESS_MANAGER.write().await;
                            pm.get_logs(&label).await
                        };
                        let result = match logs {
                            Ok((st, out, err)) => {
                                format!(
                                    "Logs for '{}':\nStatus: {:?}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}\n",
                                    label, st, out, err
                                )
                            }
                            Err(e) => format!("ERROR fetching logs: {}", e),
                        };
                        tool_results_text
                            .push_str(&format!("==== [BASH LOGS] ====\n{}\n\n", result));
                    }

                    ToolCall::BashTerminate { label } => {
                        let result = {
                            let mut pm = PROCESS_MANAGER.write().await;
                            pm.terminate(&label).await
                        };
                        let msg = match result {
                            Ok(_) => format!("Terminated background job '{}'.", label),
                            Err(e) => format!("ERROR terminating '{}': {}", label, e),
                        };
                        tool_results_text
                            .push_str(&format!("==== [BASH TERMINATE] ====\n{}\n\n", msg));
                    }
                    ToolCall::Search {
                        query,
                        count,
                        safe_search,
                    } => {
                        let result = match run_search(&query, count, safe_search).await {
                            Ok(r) => r,
                            Err(e) => format!("ERROR during search: {}", e),
                        };
                        tool_results_text.push_str(&format!(
                            "==== [SEARCH result for \"{}\"] ====\n{}\n\n",
                            query, result
                        ));
                    }
                    ToolCall::Bash { command, args } => {
                        let full_cmd = format!("{} {}", command, args.join(" "));
                        let result = match run_bash_command(&full_cmd).await {
                            Ok(r) => r,
                            Err(e) => format!("{}: {}", "BASH ERROR".red().bold(), e),
                        };
                        tool_results_text.push_str(&format!(
                            "\n{}\n{}\n",
                            "==== [BASH COMMAND RESULTS] ====".yellow().bold(),
                            result
                        ));
                    }
                    ToolCall::Scrape { url, render_js } => {
                        let result = match run_scrape(&url, render_js.unwrap_or(true)).await {
                            Ok(r) => r,
                            Err(e) => format!("ERROR during scrape: {}", e),
                        };
                        tool_results_text.push_str(&format!(
                            "==== [SCRAPE result for \"{}\"] ====\n{}\n\n",
                            url, result
                        ));
                    }
                    ToolCall::GetVar { name } => {
                        let found = session.get_var(&name);
                        let out = match found {
                            Some(value) => format!("Variable '{}': {}", name, value),
                            None => format!("Variable '{}' not found", name),
                        };
                        tool_results_text
                            .push_str(&format!("==== [GET VAR result] ====\n{}\n\n", out));
                    }
                    ToolCall::SetVar { name, value } => {
                        session.set_var(name.clone(), value.clone());
                        tool_results_text.push_str(&format!(
                            "==== [SET VAR result] ====\nSet '{}' to '{}'\n\n",
                            name, value
                        ));
                    }
                    ToolCall::ListVars => {
                        let result = session.list_vars();
                        tool_results_text
                            .push_str(&format!("==== [LIST VARS result] ====\n{}\n\n", result));
                    }
                }
            }

            if tool_results_text.len() > 500 {
                if let Ok(summary) = summarize_tool_output(&tool_results_text).await {
                    if !summary.is_empty() {
                        session.messages.push(Message::Assistant(format!(
                            "Summary of tool results:\n{}",
                            summary
                        )));
                        session.append_to_long_context(
                            "tool_summaries",
                            &json!({
                                "timestamp": chrono::Utc::now().to_rfc3339(),
                                "summary": summary
                            }),
                        );
                    }
                }
            }

            // Show assistant response
            let display_text = assistant_response.trim();
            if !display_text.is_empty() {
                println!("{} {}", "[Assistant]".blue().bold(), display_text);
            }

            if !tool_results_text.is_empty() {
                println!("{} {}", "[Tool Results]".yellow().bold(), tool_results_text);
            }
        }
    }

    Ok(())
}

//
// -----------------------------------------------------------------------------
// Main CLI Loop
// -----------------------------------------------------------------------------

/// CLI arguments
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to session file to run in test mode
    #[arg(short, long)]
    test_session: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // If test mode specified, run that instead of interactive mode
    if let Some(session_path) = args.test_session {
        return run_test_session(&session_path).await;
    }

    // The system message we always want to keep
    let preamble = r#"You are an EXPERT AI ASSISTANT. Your primary function is to assist users by answering their questions and performing tasks using a suite of TOOLS.  You are designed to handle complex, multi-step tasks, and you are equipped with tools to search the web, run bash commands, scrape web pages, and manage variables.

**Your capabilities include:**

* **Information Retrieval:** You can access and process real-time information from the internet using the `<search>` and `<scrape>` tools.
* **System Interaction:** You can interact with the underlying operating system by executing commands via the `<bash>` tool.
* **Memory Management:** You have access to both short-term variables and a long-term knowledge base to store and retrieve information across interactions.
* **Reflection and Planning:** You can use `<reflection>` blocks to think through complex problems, plan your next steps, and break down tasks into manageable parts. The system will automatically proceed after you use a `<reflection>` block.
* **Multi-Step Task Execution:** You are designed to perform tasks that require multiple steps, including tool calls, analysis of results, and iterative refinement of your approach.

**Tool Usage is MANDATORY:**

To effectively assist users, you **MUST** utilize the available tools.  Consider every user request and determine if a tool is necessary to fulfill it.  Do not attempt to answer questions or perform tasks without using tools when appropriate.

**Available Tools and How to Call Them:**

You call tools by using specific XML-like tags in your response. The system will parse these tags, execute the corresponding tools, and provide you with the results.

Here are the available tools and their usage:

1. **`<search>` Tool:**  Used for searching the web.

   * **Call Format:** `<search query="..." count="..." safe_search="..." />`
   * **Parameters:**
      * `query` (required): The search query string.  Be specific and descriptive in your queries.
      * `count` (optional, default: 5): The number of search results to return (1-10).
      * `safe_search` (optional, default: true):  Boolean value (`true` or `false`) to enable or disable safe search filtering.
   * **Example Calls:**
      * `<search query="current weather in London" />`
      * `<search query="latest cybersecurity threats" count="3" safe_search="false" />`
   * **Tool Response:** The tool will return a formatted text block containing a list of search results, including URLs and descriptions.

2. **`<bash>` Tool:** Used to execute commands in a bash shell. **Use this tool with caution and only when necessary for system-level tasks.**

   * **Call Format:** 
     - For immediate commands: `<bash command="..." args="..." />`
     - For background/long-running tasks: `<bash command="..." args="..." background="true" label="mylabel" />`
   * **Parameters:**
      * `command` (required): The base bash command to execute (e.g., `ls`, `ping`, `curl`). **Avoid potentially harmful commands like `rm`, `mv`, `dd`, `shutdown`, `reboot`, `kill`.**
      * `args` (optional): Space-separated arguments for the command.
      * `background` (optional): Set to "true" to run command in background for long-running tasks
      * `label` (required for background): Unique name to track background job
   * **Example Calls:**
      * Immediate command: `<bash command="ping" args="google.com -c 3" />`
      * Background task: `<bash command="long_running_script.sh" background="true" label="data_processing" />`
   * **Tool Response:** 
     - For immediate commands: Returns STDOUT/STDERR and exit code
     - For background tasks: Returns job ID and status
   * **Background Job Management:**
     - Check status: `<bash_status label="mylabel" />`
     - Get logs: `<bash_logs label="mylabel" />` 
     - Terminate: `<bash_terminate label="mylabel" />`

3. **`<scrape>` Tool:** Used to fetch the content of a web page.

   * **Call Format:** `<scrape url="..." render_js="..." />`
   * **Parameters:**
      * `url` (required): The URL of the web page to scrape. Ensure the URL is valid and properly formatted.
      * `render_js` (optional, default: true): Boolean value (`true` or `false`) indicating whether to render JavaScript on the page before scraping. Set to `true` for dynamic websites, `false` for static HTML.
   * **Example Calls:**
      * `<scrape url="https://www.example.com" />`
      * `<scrape url="https://news-website.com/article" render_js="false" />`
   * **Tool Response:** The tool will return the text content extracted from the HTML of the web page. For binary content, it will indicate the number of bytes received.

4. **`<get_var>` Tool:** Used to retrieve the value of a previously set variable.

   * **Call Format:** `<get_var name="..." />`
   * **Parameters:**
      * `name` (required): The name of the variable to retrieve.
   * **Example Call:** `<get_var name="current_city" />`
   * **Tool Response:** The tool will return the value of the variable if it exists, or indicate if the variable is not found.

5. **`<set_var>` Tool:** Used to store a value in a variable for later retrieval.  Variables are useful for short-term memory within a conversation.

   * **Call Format:** `<set_var name="..." value="..." />`
   * **Parameters:**
      * `name` (required): The name of the variable to set (e.g., `current_city`, `search_results_summary`). Choose descriptive names.
      * `value` (required): The value to store in the variable.
   * **Example Call:** `<set_var name="current_city" value="London" />`
   * **Tool Response:** The tool will confirm that the variable has been set.

6. **`<list_vars>` Tool:** Used to list all currently set variables and their values.

   * **Call Format:** `<list_vars />`
   * **Parameters:** None
   * **Example Call:** `<list_vars />`
   * **Tool Response:** The tool will return a list of all variables, their values, and the timestamp when they were set.

**Reflection Blocks:**

Use `<reflection>` blocks to document your thought process, plan your next steps, or outline a strategy.  This is useful for complex tasks where you need to think step-by-step.

* **Call Format:** `<reflection>Your reflection text here. Describe your thinking, planning, or analysis.</reflection>`
* **Example:** `<reflection>I need to find the capital of France. First, I will use the <search> tool to search for "capital of France". Then, I will extract the answer from the search results.</reflection>`
* **System Behavior:** The system will record your reflection but will not treat it as a tool call. It's for your internal reasoning and planning.

**Long-Term Knowledge Base (long_context):**

The `long_context` is a structured knowledge base where you can store summarized information and findings.  Tools and you can append data to it.  This knowledge is persisted across interactions and can be used to maintain context over longer conversations.  You can summarize and store important information in the `long_context` to build upon previously discovered knowledge.

**Signaling Task Completion:**

When you have finished answering the user's question or completing the requested task, and you do not need to perform any further tool calls, conclude your response with the tag `<done/>`.  This signals to the system that you have completed your chain of thought.

**Example Conversation Flow:**

**User:** What is the weather in Paris?

**Assistant:** `<reflection>The user is asking for the weather in Paris. I need to use the <search> tool to get this information.</reflection><search query="current weather in Paris" /><reflection>Now I have the search results. I will extract the current weather information and present it to the user, then I will signal that I am done.</reflection>The current weather in Paris is ... <done/>`

**Remember to:**

* **Always consider tool usage first.**
* **Use precise and descriptive queries for `<search>` and valid URLs for `<scrape>`.**
* **Be mindful of security when using `<bash>`.**
* **Utilize variables (`<set_var>`, `<get_var>`, `<list_vars>`) to manage short-term information.**
* **Use `<reflection>` for planning and complex tasks.**
* **Store summarized knowledge in the `long_context`.**
* **End your final response with `<done/>` when finished.**

By following these guidelines and effectively using the provided tools, you can be a highly effective and helpful AI assistant."#;

    // Initialize logging with log level from RUST_LOG or LOG_LEVEL environment
    let filter = EnvFilter::from_default_env();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    println!(
        "{}",
        "Welcome to the Turn-Based LLM CLI (DeepSeek)!"
            .bold()
            .bright_cyan()
    );
    println!("Type your messages and press Enter. Type '/exit' to quit.\n");

    // Load or initialize the conversation
    let mut session = load_session();
    if session.messages.is_empty() {
        // Provide an initial system role instruction if none is present
        session.messages.push(Message::System(preamble.to_string()));
    }

    // REPL loop
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let user_input = line?.trim().to_string();

        if user_input.to_lowercase() == "/exit" {
            println!("{}", "Goodbye!".green());
            break;
        } else if user_input.to_lowercase().starts_with("/forget ") {
            // Extract what to forget
            let to_forget = user_input[8..].trim();
            if to_forget.is_empty() {
                println!("{}", "Please specify what to forget".red());
                continue;
            }

            match forget_conversation(&session, to_forget).await {
                Ok(summary) => {
                    // Keep only the system message and summary
                    let system_msg = session
                        .messages
                        .iter()
                        .find(|m| matches!(m, Message::System(_)))
                        .cloned()
                        .unwrap_or(Message::System(preamble.to_string()));

                    session.messages = vec![
                        system_msg,
                        Message::Assistant(format!(
                            "(Conversation Summary - Forgot '{}')\n{}",
                            to_forget, summary
                        )),
                    ];

                    // Clear variables and long_context
                    session.variables.clear();
                    session.long_context = Value::Object(serde_json::Map::new());

                    println!(
                        "{} {}",
                        "Conversation updated to forget:".green(),
                        to_forget
                    );
                    continue;
                }
                Err(e) => {
                    eprintln!("{}: {}", "Error forgetting conversation".red(), e);
                    continue;
                }
            }
        } else if user_input.to_lowercase().starts_with("/think ") {
            // Extract what to think about
            let to_think = user_input[7..].trim();
            if to_think.is_empty() {
                println!("{}", "Please specify what to think about".red());
                continue;
            }

            // Build a temporary session with the thinking prompt
            let mut think_sess = session.clone();

            // Convert conversation to a single user message
            let mut conversation_text = String::new();
            for msg in &think_sess.messages {
                match msg {
                    Message::System(txt) => {
                        conversation_text.push_str("[System]\n");
                        conversation_text.push_str(txt);
                        conversation_text.push_str("\n\n");
                    }
                    Message::User(txt) => {
                        conversation_text.push_str("[User]\n");
                        conversation_text.push_str(txt);
                        conversation_text.push_str("\n\n");
                    }
                    Message::Assistant(txt) => {
                        conversation_text.push_str("[Assistant]\n");
                        conversation_text.push_str(txt);
                        conversation_text.push_str("\n\n");
                    }
                }
            }

            // Keep only system message and the combined conversation as user message
            think_sess
                .messages
                .retain(|m| matches!(m, Message::System(_)));
            think_sess.messages.push(Message::User(
                format!("Analyze this conversation and provide a detailed, thoughtful response to: '{}'. Consider all aspects and implications. Be thorough but concise.\n\nConversation:\n{}", to_think, conversation_text)
            ));

            match call_deepseek_api(&mut think_sess, "deepseek-reasoner").await {
                Ok(response) => {
                    println!("{} {}", "[Deep Thought]".bright_blue().bold(), response);

                    // Add the deep thought to both the main session and long_context
                    let thought = format!("(Deep Thought about '{}')\n{}", to_think, response);

                    // Add to messages
                    session.messages.push(Message::Assistant(thought.clone()));

                    // Add to long_context
                    session.append_to_long_context(
                        "deep_thoughts",
                        &json!({
                            "timestamp": chrono::Utc::now().to_rfc3339(),
                            "topic": to_think,
                            "thought": thought
                        }),
                    );
                }
                Err(e) => {
                    eprintln!("{}: {}", "Error during deep thinking".red(), e);
                    session.messages.push(Message::Assistant(format!(
                        "Error during deep thinking: {}",
                        e
                    )));
                }
            }
            continue;
        } else if user_input.to_lowercase() == "/clear" {
            // Summarize and reset the conversation
            match summarize_conversation(&session).await {
                Ok(summary) => {
                    // Keep only the system message and summary
                    let system_msg = session
                        .messages
                        .iter()
                        .find(|m| matches!(m, Message::System(_)))
                        .cloned()
                        .unwrap_or(Message::System(preamble.to_string()));

                    session.messages = vec![
                        system_msg,
                        Message::Assistant(format!("(Conversation Summary)\n{}", summary)),
                    ];

                    // Clear variables and long_context
                    session.variables.clear();
                    session.long_context = Value::Object(serde_json::Map::new());

                    println!("{}", "Conversation cleared and summarized.".green());
                    continue;
                }
                Err(e) => {
                    eprintln!("{}: {}", "Error summarizing conversation".red(), e);
                    continue;
                }
            }
        }

        // 1) Add the user's message to the conversation and show it
        println!("{} {}", "[User]".green().bold(), user_input);
        session.messages.push(Message::User(user_input.clone()));

        // Add context as assistant messages
        let recent_vars = session.recent_vars();
        let knowledge_summary = session.summarize_long_context();

        if !recent_vars.is_empty() {
            session.messages.push(Message::Assistant(format!(
                "Recent variables:\n{}",
                recent_vars
            )));
        }
        if !knowledge_summary.is_empty() {
            session.messages.push(Message::Assistant(knowledge_summary));
        }

        // 2) Call the LLM with a simple timeout, always using "deepseek-chat" for normal conversation
        let result = timeout(
            Duration::from_secs(300),
            call_deepseek_api(&mut session, "deepseek-chat"),
        )
        .await;

        let assistant_response = match result {
            Ok(Ok(content)) => content,
            Ok(Err(e)) => {
                eprintln!("{}: {}", "[Error from LLM]".red(), e.to_string());
                session
                    .messages
                    .push(Message::Assistant(format!("Error from LLM: {e}")));
                continue;
            }
            Err(_) => {
                eprintln!("{}", "[Timed out waiting for LLM]".red());
                session
                    .messages
                    .push(Message::Assistant("DeepSeek API timed out.".into()));
                continue;
            }
        };

        // 3) Parse any reflection blocks
        let reflection_blocks = extract_reflection_blocks(&assistant_response);
        for block in reflection_blocks {
            session
                .messages
                .push(Message::Assistant(format!("(reflection)\n{}", block)));
        }

        // 4) Parse tool calls
        let tool_calls = parse_tool_calls(&assistant_response);
        let mut tool_results_text = String::new();
        for call in tool_calls {
            match call {
                ToolCall::BashBackground {
                    label,
                    command,
                    args,
                } => {
                    let full_cmd = format!("{} {}", command, args.join(" "));
                    let result = match {
                        let mut pm = PROCESS_MANAGER.write().await;
                        pm.spawn_process(&label, &full_cmd).await
                    } {
                        Ok(_) => format!(
                            "Spawned background job '{}' with command: {}",
                            label, full_cmd
                        ),
                        Err(e) => format!("ERROR spawning background job '{}': {}", label, e),
                    };
                    tool_results_text
                        .push_str(&format!("==== [BASH BACKGROUND] ====\n{}\n\n", result));
                }

                ToolCall::BashStatus { label } => {
                    let status = {
                        let mut pm = PROCESS_MANAGER.write().await;
                        pm.refresh_status(&label).await
                    };
                    let result = match status {
                        Ok(st) => format!("Status for '{}': {:?}", label, st),
                        Err(e) => format!("ERROR checking status: {}", e),
                    };
                    tool_results_text.push_str(&format!("==== [BASH STATUS] ====\n{}\n\n", result));
                }

                ToolCall::BashLogs { label } => {
                    let logs = {
                        let mut pm = PROCESS_MANAGER.write().await;
                        pm.get_logs(&label).await
                    };
                    let result = match logs {
                        Ok((st, out, err)) => {
                            format!(
                                "Logs for '{}':\nStatus: {:?}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}\n",
                                label, st, out, err
                            )
                        }
                        Err(e) => format!("ERROR fetching logs: {}", e),
                    };
                    tool_results_text.push_str(&format!("==== [BASH LOGS] ====\n{}\n\n", result));
                }

                ToolCall::BashTerminate { label } => {
                    let result = {
                        let mut pm = PROCESS_MANAGER.write().await;
                        pm.terminate(&label).await
                    };
                    let msg = match result {
                        Ok(_) => format!("Terminated background job '{}'.", label),
                        Err(e) => format!("ERROR terminating '{}': {}", label, e),
                    };
                    tool_results_text.push_str(&format!("==== [BASH TERMINATE] ====\n{}\n\n", msg));
                }
                ToolCall::Search {
                    query,
                    count,
                    safe_search,
                } => {
                    let result = match run_search(&query, count, safe_search).await {
                        Ok(r) => r,
                        Err(e) => format!("ERROR during search: {}", e),
                    };
                    tool_results_text.push_str(&format!(
                        "==== [SEARCH result for \"{}\"] ====\n{}\n\n",
                        query, result
                    ));
                }
                ToolCall::Bash { command, args } => {
                    let full_cmd = format!("{} {}", command, args.join(" "));
                    let result = match run_bash_command(&full_cmd).await {
                        Ok(r) => r,
                        Err(e) => format!("{}: {}", "BASH ERROR".red().bold(), e),
                    };
                    tool_results_text.push_str(&format!(
                        "{}\n{}\n",
                        "==== [BASH COMMAND RESULTS] ====".yellow().bold(),
                        result
                    ));
                }
                ToolCall::Scrape { url, render_js } => {
                    let result = match run_scrape(&url, render_js.unwrap_or(true)).await {
                        Ok(r) => r,
                        Err(e) => format!("ERROR during scrape: {}", e),
                    };
                    tool_results_text.push_str(&format!(
                        "==== [SCRAPE result for \"{}\"] ====\n{}\n\n",
                        url, result
                    ));
                }
                ToolCall::GetVar { name } => {
                    let found = session.get_var(&name);
                    let out = match found {
                        Some(value) => format!("Variable '{}': {}", name, value),
                        None => format!("Variable '{}' not found", name),
                    };
                    tool_results_text.push_str(&format!("==== [GET VAR result] ====\n{}\n\n", out));
                }
                ToolCall::SetVar { name, value } => {
                    session.set_var(name.clone(), value.clone());
                    tool_results_text.push_str(&format!(
                        "==== [SET VAR result] ====\nSet '{}' to '{}'\n\n",
                        name, value
                    ));
                }
                ToolCall::ListVars => {
                    let result = session.list_vars();
                    tool_results_text
                        .push_str(&format!("==== [LIST VARS result] ====\n{}\n\n", result));
                }
            }
        }

        // 5) Summarize large tool output automatically
        if tool_results_text.len() > 500 {
            if let Ok(summary) = summarize_tool_output(&tool_results_text).await {
                if !summary.is_empty() {
                    session.messages.push(Message::Assistant(format!(
                        "Summary of tool results:\n{}",
                        summary
                    )));
                    session.append_to_long_context(
                        "tool_summaries",
                        &json!({
                            "timestamp": chrono::Utc::now().to_rfc3339(),
                            "summary": summary
                        }),
                    );
                }
            }
        }

        // If we had tool results, we can place them in conversation if they're not too large
        if !tool_results_text.is_empty() {
            if tool_results_text.len() <= MAX_TOOL_RESULT_SIZE {
                session.messages.push(Message::Assistant(format!(
                    "Tool results:\n{}",
                    tool_results_text
                )));
            } else {
                session.messages.push(Message::Assistant(
                    "Tool results too large to store in session".to_string(),
                ));
            }
        }

        // 6) Add the assistant's raw response to the conversation
        let final_response = assistant_response.clone();
        session
            .messages
            .push(Message::Assistant(final_response.clone()));

        // Display assistant response and tool results
        if !final_response.is_empty() {
            println!("{} {}", "[Assistant]".blue().bold(), final_response);
        }
        if !tool_results_text.is_empty() {
            println!("{} {}", "[Tool Results]".yellow().bold(), tool_results_text);
        }

        // Save session
        if let Err(e) = save_session(&session) {
            eprintln!("Failed to save session: {}", e);
        }
    }

    Ok(())
}
