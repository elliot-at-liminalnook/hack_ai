# Turn-Based LLM CLI (DeepSeek)

This project is a command-line interface (CLI) application designed to interact with a language model for various tasks. It supports executing commands, searching the web, scraping web pages, and managing variables, all while maintaining a conversation session.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   Ensure you have Rust and Cargo installed. Then, run:
   ```bash
   cargo build
   ```

3. **Set up environment variables**:
   - `DEEPSEEK_API_KEY`: Your API key for accessing the DeepSeek service.
   - `BRAVE_SEARCH_API_KEY`: Your API key for Brave Search.
   - `SCRAPINGBEE_API_KEY`: Your API key for ScrapingBee.

## Usage

To start the application, run:
```bash
cargo run
```

### Commands

- **/exit**: Exit the application.
- **/remember <text>**: Save a memory to `memories.txt`.
- **/forget <text>**: Forget specific information from the conversation.
- **/think <text>**: Perform a deep analysis on the conversation.
- **/run <command>**: Run a bash command in the background.
- **/add <file_path>**: Add the contents of a file to the conversation.
- **/clear**: Clear the conversation and summarize it.
- **/eager**: Check the status of all background jobs immediately.

## Features

- **Information Retrieval**: Search the web and scrape web pages.
- **System Interaction**: Execute bash commands.
- **Memory Management**: Store and retrieve variables.
- **Reflection and Planning**: Use reflection blocks for complex tasks.
- **Multi-Step Task Execution**: Handle tasks requiring multiple steps.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
