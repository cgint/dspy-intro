# Agent Coding Assistant

This project is an **advanced streaming AI agent** designed to act as a **Coding Assistant**. Built using **DSPy + FastAPI + Socket.IO**, it not only provides expertise on topics like Google Ads but can also **interact directly with the codebase**, including reading/writing files, executing shell commands, and performing code analysis. It offers real-time streaming responses with comprehensive tool transparency and grounding information.

## ✨ Features

- **Real-time Streaming**: WebSocket-based streaming responses
- **Tool Transparency**: See which tools are being used (e.g., internal knowledge, web search, filesystem)
- **Grounding Information**: Track sources, references, and queries in real-time
- **Dual Interfaces**: Both HTTP REST API and WebSocket for different use cases
- **Error Handling**: Graceful error handling and recovery
- **Google Ads Expertise**: Specialized knowledge for campaign setup, optimization, and strategies
- **Internal Knowledge Base**: Access to project-specific documentation
- **Web Search Integration**: Up-to-date information retrieval via Tavily
- **Codebase Interaction**: Read, list, and write files directly within the project
- **Restricted Shell Execution**: Safely run allowlisted shell commands (git, uv, pytest, ls, etc.)
- **Codebase Analysis (Codegiant)**: Semantic Q&A over the entire codebase
- **Git Diff Review (Codegiant)**: Review uncommitted code changes
- **Code Term Search**: Efficiently search for terms/patterns across files
- **Direct Gemini Interaction**: Leverage Google Gemini CLI for arbitrary prompts

## 🚀 Setup

This project uses `uv` for dependency management.

1.  **Install dependencies:**
    ```bash
    uv sync
    ```

2.  **Set up your environment:** 
    You'll need API keys for Google Gemini and Tavily search. Ensure `gcloud` is authenticated for Gemini CLI tools.
    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    export TAVILY_API_KEY="your-tavily-api-key"
    # Ensure gcloud is authenticated for Gemini CLI tools
    # gcloud auth application-default login
    ```

## 🎯 Running the Application

### Option 1: Web Interface (recommended)
```bash
uv run python web/start_web_server.py
```
**Access at**: `http://localhost:8000/`

### Option 2: API server with Socket.IO
```bash
uv run python start_server.py
```

### Option 3: Direct uvicorn
```bash
uv run uvicorn api.main:socket_app --host 0.0.0.0 --port 8000 --reload
```

### Option 4: Console-only mode (original)
```bash
uv run python dspy_agent_refactored.py
```

## 🌐 API Endpoints

- **Root**: `http://localhost:8000/` - API information
- **Health Check**: `http://localhost:8000/health` - Service health
- **Documentation**: `http://localhost:8000/docs` - Interactive API docs
- **Socket.IO Info**: `http://localhost:8000/socket.io-info` - WebSocket details

### HTTP Endpoints
- `POST /api/v1/ask` - Ask question (streaming service)
- `POST /api/v1/ask-sync` - Ask question (original service)
- `GET /api/v1/capabilities` - Get AI capabilities and available tools

### WebSocket Events
- **Client → Server**:
    - `ask_question`: Submit a new question to the agent.
    - `cancel_question`: Cancel the current question processing.
    - `get_session_info`: Request information about the current session.
    - `load_session_history`: Request the chat history for the current session.
    - `ping`: Send a ping to check connection status.
- **Server → Client**:
    - `connection_confirmed`: Confirms successful WebSocket connection.
    - `question_start`: Signals the start of processing a new question.
    - `tool_start`: Indicates a tool has started execution.
    - `tool_progress`: Provides updates on long-running tool execution.
    - `tool_complete`: Signals a tool has completed successfully.
    - `tool_error`: Reports an error during tool execution.
    - `answer_chunk`: Streams parts of the AI's answer in real-time.
    - `grounding_update`: Provides real-time updates on sources and queries used.
    - `answer_complete`: Signals the AI's final answer is complete.
    - `question_cancelled`: Confirms that question processing was cancelled.
    - `session_info`: Provides requested session information.
    - `session_history_loaded`: Returns the chat history for the session.
    - `error`: Reports general errors from the server.
    - `pong`: Response to a client `ping`.

## 🧪 Testing

Test the WebSocket integration:
```bash
uv run python test_websocket_client.py
```

## 📁 Project Structure

Created with
```bash
tree|egrep "(\.py|web$|api$)"|grep -v "\.cpy"|grep -v "__init__"|grep -v "__pycache__"
```

```
├── api
│   ├── main.py
│   │   ├── agent.py
│   │   └── health.py
│   └── websocket_manager.py
├── chat_history_converter.py
├── cli_gather_info.py
├── dspy_agent_classifier_credentials_passwords_examples.py
├── dspy_agent_classifier_credentials_passwords_optimized.py
├── dspy_agent_classifier_credentials_passwords.py
├── dspy_agent_expert_ai.py
├── dspy_agent_lm_vertexai.py
├── dspy_agent_streaming_service.py
├── dspy_agent_tool_cgiant.py
├── dspy_agent_tool_code_term_search.py
├── dspy_agent_tool_gemini.py
├── dspy_agent_tool_internal_knowledge.py
├── dspy_agent_tool_lc_filesystem.py
├── dspy_agent_tool_restricted_shell.py
├── dspy_agent_tool_rm_google_search_info.py.txt
├── dspy_agent_tool_rm_tavily.py
├── dspy_agent_tool_streaming_internal_knowledge.py
├── dspy_agent_tool_streaming_websearch_tavily.py
├── dspy_agent_tool_websearch_tavily.py
├── dspy_agent_util_grounding_manager.py
├── dspy_agent_util_streaming_grounding_manager.py
├── dspy_constants.py
├── dspy_pricing_service.py
├── session_history_manager.py
├── session_models.py
├── session_storage.py
├── start_server.py
├── test_websocket_client.py
└── web
    ├── start_web_server.py
```