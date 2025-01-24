use anyhow::{anyhow, Result};
use colored::*;
use html2text::from_read;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use clap::Parser;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use url::Url;

pub fn extract_text_from_html(html: &str, url: Option<&str>) -> String {
    let mut text = from_read(html.as_bytes(), 80).expect("Failed to parse HTML");

    // If URL provided, add it as reference
    if let Some(url_str) = url {
        if let Ok(parsed_url) = Url::parse(url_str) {
            text.push_str(&format!("\n\nSource: {}", url_str));

            if let Some(domain) = parsed_url.domain() {
                text.push_str(&format!("\nDomain: {}", domain));
            }
        }
    }

    text
}

//
// -----------------------------------------------------------------------------
// 1. Data Structures: Conversation, Messages, Tool Calls
// -----------------------------------------------------------------------------

/// Represents a single message in the conversation: system, user, or assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", content = "content")]
pub enum Message {
    System(String),
    User(String),
    Assistant(String),
}

/// The entire conversation session, which we store in JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSession {
    pub messages: Vec<Message>,
    pub variables: std::collections::HashMap<String, String>,
}

impl ConversationSession {
    pub fn new() -> Self {
        Self { 
            messages: vec![],
            variables: std::collections::HashMap::new(),
        }
    }

    /// Get a variable value by key
    pub fn get_var(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }

    /// Set a variable value
    pub fn set_var(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }

    /// List all variables
    pub fn list_vars(&self) -> String {
        if self.variables.is_empty() {
            return "No variables stored".to_string();
        }
        self.variables.iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Simple wrapper to highlight possible tool calls found in the assistant response.
/// We parse them with regex. Example blocks:
/// <search>some query</search>
/// <bash>some bash command</bash>
/// <scrape url>https://example.com</scrape url>
#[derive(Debug)]
pub enum ToolCall {
    Search {
        query: String,
        count: Option<u8>,
        safe_search: Option<bool>
    },
    Bash {
        command: String,
        args: Vec<String>
    },
    Scrape {
        url: String,
        render_js: Option<bool>
    },
    GetVar {
        name: String
    },
    SetVar {
        name: String,
        value: String
    },
    ListVars
}

//
// -----------------------------------------------------------------------------
// 2. Minimal Tool Implementations (Search, Bash, Scrape)
//    We’ll use simplified “defaults” for each.
// -----------------------------------------------------------------------------

// -- 2.1 Bash Tool
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
        result.push_str(&format!("\n{}:\n{}", "STDERR".red().bold(), stderr));
    }

    Ok(result)
}

// -- 2.2 Search Tool (using the Brave Search snippet you provided, but simplified)
async fn run_search(query: &str, count: Option<u8>, safe_search: Option<bool>) -> Result<String> {
    // For demonstration, we’ll do a simple GET to Brave Search API or
    // you can replace it with a mock or actual logic.
    //
    // In real usage, put your Brave API key in an environment variable: BRAVE_SEARCH_API_KEY
    let api_key = env::var("BRAVE_SEARCH_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        return Ok(
            "No BRAVE_SEARCH_API_KEY found. Provide one to enable real searching.".to_string(),
        );
    }

    let client = reqwest::Client::new();
    let url = "https://api.search.brave.com/res/v1/web/search";

    // Basic parameters
    let count = count.unwrap_or(5);
    let safe = safe_search.unwrap_or(true);
    let safesearch = if safe { "strict" } else { "off" };
    let params = [
        ("q", query),
        ("count", &count.to_string()),
        ("safesearch", safesearch)
    ];

    // Build the request
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

    // Parse JSON and extract relevant info
    let json_response: Value = resp.json().await?;
    let mut result = String::new();
    result.push_str(&format!("Search Query: {}\n\n", query));

    if let Some(web_results) = json_response.get("web").and_then(|w| w.get("results")) {
        if let Some(results) = web_results.as_array() {
            for (i, item) in results.iter().enumerate() {
                if i >= 5 { break; } // Limit to top 5 results
                
                if let (Some(url), Some(desc)) = (item.get("url"), item.get("description")) {
                    if let (Some(url_str), Some(desc_str)) = (url.as_str(), desc.as_str()) {
                        result.push_str(&format!("• {}: {}\n\n", 
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

// -- 2.3 Scrape Tool (using simplified ScrapingBee logic)
async fn run_scrape(url: &str, render_js: bool) -> Result<String> {
    // For real usage, put your ScrapingBee API key in SCRAPINGBEE_API_KEY
    let api_key = env::var("SCRAPINGBEE_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        return Ok(
            "No SCRAPINGBEE_API_KEY found. Provide one to enable real scraping.".to_string(),
        );
    }

    let client = reqwest::Client::new();
    let base_url = "https://app.scrapingbee.com/api/v1/";

    let resp = client
        .get(base_url)
        .query(&[
            ("api_key", &api_key),
            ("url", &url.to_string()),
            ("render_js", &render_js.to_string()),
        ])
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await?;
        return Err(anyhow!("ScrapingBee error: {text}"));
    }

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type.contains("text") || content_type.contains("json") {
        let text = resp.text().await?;
        let readable_text = extract_text_from_html(&text, Some(url));
        let result = format!("Scraped URL: {}\nExtracted Text:\n{}", url, readable_text);
        Ok(result)
    } else {
        let bytes = resp.bytes().await?;
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
// 3. LLM Call: DeepSeek Chat (OpenAI-compatible format)
// -----------------------------------------------------------------------------

/// We convert our internal conversation to the JSON format that DeepSeek expects.
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

/// Perform a single chat completion call to DeepSeek with the entire conversation.
async fn call_deepseek_api(session: &ConversationSession) -> Result<String> {
    let api_key = env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        error!("Missing DEEPSEEK_API_KEY environment variable!");
        return Err(anyhow!("DEEPSEEK_API_KEY not set"));
    }

    // Convert internal session messages -> DeepSeekMessage
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
        model: "deepseek-chat".to_string(),
        messages: ds_messages,
        stream: false,
        temperature: 0.7,
    };

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

    // The response body should match an OpenAI-like structure
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
}

//
// -----------------------------------------------------------------------------
// 4. Parsing the Assistant's Tool Calls (<search>, <bash>, <scrape url>, etc.)
// -----------------------------------------------------------------------------

/// For the MVP, we use a few regexes to find all tool calls in the assistant’s text.
/// e.g. <search>some query</search>  
/// We do this in a naive, repeated way. Each match includes the entire snippet.
fn parse_tool_calls(assistant_text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    
    // Parse search tool calls
    let re_search = Regex::new(r#"(?s)<search\s+query="([^"]+)"(?:\s+count="(\d+)"|\s+safe_search="(true|false)")*\s*/>"#).unwrap();
    for cap in re_search.captures_iter(assistant_text) {
        let query = cap[1].trim().to_string();
        let count = cap.get(2).and_then(|m| m.as_str().parse::<u8>().ok());
        let safe_search = cap.get(3).and_then(|m| m.as_str().parse::<bool>().ok());
        calls.push(ToolCall::Search { query, count, safe_search });
    }

    // Parse bash tool calls
    let re_bash = Regex::new(r#"(?s)<bash\s+command="([^"]+)"(?:\s+args="([^"]+)")*\s*/>"#).unwrap();
    for cap in re_bash.captures_iter(assistant_text) {
        let command = cap[1].trim().to_string();
        let args = cap.get(2)
            .map(|m| m.as_str().split_whitespace().map(|s| s.to_string()).collect())
            .unwrap_or_else(Vec::new);
        calls.push(ToolCall::Bash { command, args });
    }

    // Parse scrape tool calls
    let re_scrape = Regex::new(r#"(?s)<scrape\s+url="([^"]+)"(?:\s+render_js="(true|false)")*\s*/>"#).unwrap();
    for cap in re_scrape.captures_iter(assistant_text) {
        let url = cap[1].trim().to_string();
        let render_js = cap.get(2).and_then(|m| m.as_str().parse::<bool>().ok());
        calls.push(ToolCall::Scrape { url, render_js });
    }

    // Parse get var tool calls
    let re_get_var = Regex::new(r#"(?s)<get_var\s+name="([^"]+)"\s*/>"#).unwrap();
    for cap in re_get_var.captures_iter(assistant_text) {
        let name = cap[1].trim().to_string();
        calls.push(ToolCall::GetVar { name });
    }

    // Parse set var tool calls
    let re_set_var = Regex::new(r#"(?s)<set_var\s+name="([^"]+)"\s+value="([^"]+)"\s*/>"#).unwrap();
    for cap in re_set_var.captures_iter(assistant_text) {
        let name = cap[1].trim().to_string();
        let value = cap[2].trim().to_string();
        calls.push(ToolCall::SetVar { name, value });
    }

    // Parse list vars tool calls
    let re_list_vars = Regex::new(r"(?s)<list_vars\s*/>").unwrap();
    if re_list_vars.is_match(assistant_text) {
        calls.push(ToolCall::ListVars);
    }

    calls
}

/// Remove the tool blocks from the assistant’s textual response
/// so the final “display text” for the user is cleaned up.
fn remove_tool_blocks(assistant_text: &str) -> String {
    let re_tool_blocks =
        Regex::new(r"(?s)<(search|bash|scrape url)>(.*?)</(search|bash|scrape url)>").unwrap();
    re_tool_blocks.replace_all(assistant_text, "").to_string()
}

//
// -----------------------------------------------------------------------------
// 5. Session Persistence
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
        match File::open(path) {
            Ok(file) => {
                if let Ok(sess) = serde_json::from_reader(file) {
                    return sess;
                }
            }
            Err(_) => {}
        }
    }
    ConversationSession::new()
}

//
// -----------------------------------------------------------------------------
// 6. Test Mode - Run a session file and show results
// -----------------------------------------------------------------------------

/// Run a session file in test mode - process all messages and show results
async fn run_test_session(session_path: &str) -> Result<()> {
    let path = PathBuf::from(session_path);
    let session = load_session_from_path(&path);
    
    // Process each user message in sequence
    for msg in &session.messages {
        if let Message::User(user_input) = msg {
            println!("{} {}", "[User]".green().bold(), user_input);
            
            // Call the LLM
            let result = timeout(Duration::from_secs(30), call_deepseek_api(&session)).await;
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

            // Parse and run tool calls
            let tool_calls = parse_tool_calls(&assistant_response);
            let mut tool_results_text = String::new();
            for call in tool_calls {
                match call {
                    ToolCall::Search(q) => {
                        let result = match run_search(&q).await {
                            Ok(r) => r,
                            Err(e) => format!("ERROR during search: {}", e),
                        };
                        tool_results_text.push_str(&format!(
                            "==== [SEARCH result for \"{}\"] ====\n{}\n\n",
                            q, result
                        ));
                    }
                    ToolCall::Bash(cmd) => {
                        let result = match run_bash_command(&cmd).await {
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
                    ToolCall::GetVar(var_name) => {
                        let result = match session.get_var(&var_name) {
                            Some(value) => format!("Variable '{}': {}", var_name, value),
                            None => format!("Variable '{}' not found", var_name),
                        };
                        tool_results_text.push_str(&format!(
                            "==== [GET VAR result] ====\n{}\n\n",
                            result
                        ));
                    }
                    ToolCall::SetVar(var_expr) => {
                        let parts: Vec<&str> = var_expr.splitn(2, '=').collect();
                        if parts.len() == 2 {
                            let var_name = parts[0].trim().to_string();
                            let var_value = parts[1].trim().to_string();
                            session.set_var(var_name.clone(), var_value.clone());
                            tool_results_text.push_str(&format!(
                                "==== [SET VAR result] ====\nSet '{}' to '{}'\n\n",
                                var_name, var_value
                            ));
                        } else {
                            tool_results_text.push_str(&format!(
                                "==== [SET VAR ERROR] ====\nInvalid format. Use: <set var>name=value</set var>\n\n"
                            ));
                        }
                    }
                    ToolCall::ListVars => {
                        let result = session.list_vars();
                        tool_results_text.push_str(&format!(
                            "==== [LIST VARS result] ====\n{}\n\n",
                            result
                        ));
                    }
                }
            }

            // Show assistant response and tool results
            let display_text = &assistant_response.trim().to_string();
            if !display_text.is_empty() {
                println!(
                    "{} {}",
                    "[Assistant]".blue().bold(),
                    display_text
                );
            }
            
            if !tool_results_text.is_empty() {
                println!(
                    "{} {}",
                    "[Tool Results]".yellow().bold(),
                    tool_results_text
                );
            }
        }
    }
    
    Ok(())
}

//
// -----------------------------------------------------------------------------
// 7. Main CLI Loop
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


    let preamble = r#"You are an EXPERT AI ASSISTANT whose PRIMARY FUNCTION is to use tools to answer user questions.  You are **required** to use tools whenever they can provide a more accurate, up-to-date, or comprehensive answer than your internal knowledge.

You have access to the following **essential tools**:

* `<search query="query" [count="5"] [safe_search="true"] />`: **YOU MUST USE THIS TOOL WHENEVER THE USER ASKS A QUESTION THAT REQUIRES INFORMATION FROM THE INTERNET, CURRENT EVENTS, FACTS, OR SPECIFIC WEB CONTENT.** Example: `<search query="current weather in Paris" count="3" safe_search="true" />`

* `<bash command="command" [args="arg1 arg2"] />`: Use this tool to execute safe bash commands for system-related tasks. Example: `<bash command="ls" args="-l -a" />`

* `<scrape url="URL" [render_js="true"] />`: Use this tool to get the text content of webpages. Example: `<scrape url="https://example.com" render_js="true" />`

* **Variable Tools:**
  - `<set_var name="name" value="value" />`: Store important information for later use. Example: `<set_var name="target_ip" value="192.168.1.100" />`
  - `<get_var name="name" />`: Retrieve a stored variable. Example: `<get_var name="target_ip" />`
  - `<list_vars />`: List all stored variables and their values

**YOUR OPERATING PROCEDURE IS AS FOLLOWS:**

1. **Analyze User Request:** Understand what the user is asking.
2. **IMMEDIATELY CONSIDER TOOL USE:**  **For almost every question, your first thought MUST be: "Can a tool help me answer this better?"**
3. **PRIORITIZE SEARCH:** **If the question relates to facts, current events, or anything that might be on the internet, YOU MUST USE the `<search>` tool FIRST.**  Do not rely on your internal knowledge for these types of questions.
4. **Use `<bash>` or `<scrape>` if relevant:**  If the user's request implies a system command or scraping a webpage, use those tools accordingly.
5. **If NO tool is genuinely needed (very rare for information-seeking questions), then and ONLY then answer directly.**
6. **Respond clearly:** Tell the user what you are doing and if you used a tool. Example: "I am using the `<search>` tool to find..." or "Based on my search results..."

**YOU ARE DESIGNED TO BE A TOOL-USING AI.  YOUR VALUE COMES FROM YOUR ABILITY TO USE TOOLS EFFECTIVELY.  DO NOT HESITATE TO USE TOOLS. USE THEM PROACTIVELY AND FREQUENTLY.**

Note that you can use multiple tools in one response: 

<search> what's the weather in orange county, TX </search> <search> Is the orange county texas basketball game still on for tonight </search>  <search> What's the best taco to eat during a basketball game? </search> 

**Example of REQUIRED tool use:**

User: "What is the weather in Paris right now?"
Assistant: `<search>current weather in Paris</search>`  **<-- CORRECT - TOOL USED**
Assistant: "The weather in Paris is sunny." **<-- INCORRECT - ANSWERING WITHOUT TOOL WHEN TOOL IS NEEDED**

User: "List files in my home directory."
Assistant: `<bash>ls ~</bash>` **<-- CORRECT - TOOL USED**

User: "The target IP is 192.168.1.100"
Assistant: `<set var>target_ip=192.168.1.100</set var>` **<-- CORRECT - STORING IMPORTANT INFO**

User: "What IP are we targeting?"
Assistant: `<get var>target_ip</get var>` **<-- CORRECT - RETRIEVING STORED INFO**

**FAILURE TO USE TOOLS WHEN APPROPRIATE IS A CRITICAL ERROR.**  Ensure you are always checking if a tool can improve your answer before responding from internal knowledge. Use variables to store and retrieve important information across multiple turns of conversation."#;

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
        // You can provide an initial system role instruction if you want
        session.messages.push(Message::System(
            preamble.to_string()
        ));
    }

    // REPL loop
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let user_input = line?.trim().to_string();

        if user_input.to_lowercase() == "/exit" {
            println!("{}", "Goodbye!".green());
            break;
        }

        // 1) Add the user’s message to the conversation and show it
        println!("{} {}", "[User]".green().bold(), user_input);
        session.messages.push(Message::User(user_input.clone()));

        // 2) Call the LLM
        // (with a simple timeout to avoid indefinite hang - 30s for demo)
        let result = timeout(Duration::from_secs(30), call_deepseek_api(&session)).await;

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

        // 3) Parse out any tool calls
        let tool_calls = parse_tool_calls(&assistant_response);

        // 4) If there are tool calls, we run them all in sequence, gather results,
        //    then append them to the conversation so the LLM can “see” those results.
        let mut tool_results_text = String::new();
        for call in tool_calls {
            match call {
                ToolCall::Search { query, count, safe_search } => {
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
                    let result = match session.get_var(&name) {
                        Some(value) => format!("Variable '{}': {}", name, value),
                        None => format!("Variable '{}' not found", name),
                    };
                    tool_results_text.push_str(&format!(
                        "==== [GET VAR result] ====\n{}\n\n",
                        result
                    ));
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
                    tool_results_text.push_str(&format!(
                        "==== [LIST VARS result] ====\n{}\n\n",
                        result
                    ));
                }
            }
        }

        // If we had tool calls, we now add one more “assistant” message
        // so the conversation chain includes the results. The LLM can then reference them.
        if !tool_results_text.is_empty() {
            let tool_msg = format!("Tool results:\n{}", tool_results_text);
            session.messages.push(Message::Assistant(tool_msg));

            // Optionally, call the LLM again so it can comment on the tool results.
            // But for an MVP, we might let the user read the tool results first,
            // then the user asks for next steps.
            //
            // In this example, we do NOT automatically call the LLM again
            // because we’re “turn-based.” But you could if desired.
        }

        // 5) Show the assistant’s textual response to the user (minus the <tool> blocks)
        let display_text = &assistant_response.trim().to_string();
        if !display_text.is_empty() {
            // Save it to conversation
            session
                .messages
                .push(Message::Assistant(display_text.clone()));

            // Print it colorized
            println!(
                "{} {}",
                "[Assistant]".blue().bold(),
                display_text
            );
        }

        // 6) Save the updated session
        if let Err(e) = save_session(&session) {
            eprintln!("Failed to save session: {}", e);
        }
    }

    Ok(())
}

