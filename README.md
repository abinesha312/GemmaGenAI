# 🎓 UNT Multi-Agent System

A sophisticated multi-agent system for the University of North Texas (UNT) that provides specialized assistance for various academic tasks using the 🧠 Gemma 3 27B model.

## 📌 Overview

This project implements a multi-agent system with specialized agents for different academic tasks:

- 📧 **Email Composition Agent**: Helps draft professional academic emails
- 📑 **Research Paper Agent**: Assists with research paper composition and analysis
- 📚 **Academic Concepts Agent**: Explains academic concepts and theories
- 🔗 **Redirect Agent**: Directs users to appropriate UNT resources
- 🏫 **General Agent**: Handles general queries about UNT

The system uses a sophisticated classification mechanism to determine which agent should handle a user query, and each agent follows a structured, step-by-step reasoning approach to provide comprehensive responses.

## 🚀 Features

- 🤖 **Specialized Agents**: Each agent is tailored to a specific academic task
- 🎯 **Intelligent Classification**: Uses TF-IDF vectorization and cosine similarity to classify user queries
- 📝 **Structured Responses**: All agents provide well-formatted, comprehensive responses
- 🛠 **Pydantic Models**: Input validation and structured data handling
- 💻 **Chainlit Interface**: Modern web UI for interacting with the agents
- ⚡ **vLLM Integration**: Efficient inference using the Gemma 3 27B model

## 📁 Project Structure

```
.
├── 🐳 Dockerfile              # Container definition for the main application
├── 🐳 Dockerfile.vllm         # Container definition for the vLLM server
├── 📄 requirements.txt        # Python dependencies
├── 🛠 run_podman.sh           # Script to build and run the application with Podman
└── 📂 src/
    ├── 🚀 app.py              # Main application entry point
    ├── 🤖 agents/
    │   ├── 🔧 base_agent.py   # Base agent class with common functionality
    │   ├── 📌 registry.py     # Agent registry and classification
    │   └── 🎯 specialized_agents.py  # Specialized agent implementations
    ├── ⚙️ config/
    │   ├── 📝 prompts.py      # Agent prompts and templates
    │   └── 🔧 settings.py     # Application settings and configuration
    └── 📂 models/
        ├── 📊 classification.py  # Query classification system
        └── ✅ query_models.py    # Pydantic models for query validation
```

## 🛠 Prerequisites

- 🐍 Python 3.10+
- 🛢 Podman
- 🍺 Homebrew (for Linux)
- 🎮 CUDA-capable GPU (for vLLM server)

## 📥 Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install Podman using Homebrew:

   ```bash
   brew install podman
   ```

3. Initialize Podman (if needed):
   ```bash
   podman machine init
   podman machine start
   ```

## ▶️ Running the Application

1. Make the run script executable:

   ```bash
   chmod +x run_podman.sh
   ```

2. Run the application:

   ```bash
   ./run_podman.sh
   ```

3. Access the application at `http://localhost:8000`

## 🔧 Environment Variables

The application uses the following environment variables:

- 🏷 `MODEL_ID`: The model ID to use (default: "google/gemma-3-27b-it")
- 🌐 `INFERENCE_SERVER_URL`: URL of the vLLM server (default: "http://vllm-server:5000/v1")
- 🔄 `MAX_RETRIES`: Maximum number of retries for API calls (default: 3)
- ⏳ `RETRY_DELAY`: Delay between retries in seconds (default: 2)
- ⏱ `REQUEST_TIMEOUT`: Timeout for API requests in seconds (default: 30)
- 🌍 `CHAINLIT_HOST`: Host for the Chainlit server (default: "0.0.0.0")
- 📡 `CHAINLIT_PORT`: Port for the Chainlit server (default: 8000)
- 📜 `LOG_LEVEL`: Logging level (default: "INFO")

## 🤖 Agent Capabilities

### 📧 Email Composition Agent

- Drafts professional academic emails
- Follows proper email structure and formatting
- Maintains appropriate tone and style
- Handles common scenarios like extension requests and meeting scheduling

### 📑 Research Paper Agent

- Helps with research paper planning and structure
- Provides guidance on research methodology
- Ensures proper academic writing standards
- Supports various citation styles and formats

### 📚 Academic Concepts Agent

- Explains academic concepts and theories
- Adapts explanations to different difficulty levels
- Provides learning support and resources
- Covers various subject areas

### 🔗 Redirect Agent

- Directs users to relevant UNT resources
- Provides detailed information about available services
- Includes direct links to resources
- Offers contact information and usage guidelines

## 🏗 Development

To modify or extend the system:

1. Update the agent prompts in `src/config/prompts.py`
2. Modify the agent implementations in `src/agents/specialized_agents.py`
3. Adjust the classification system in `src/models/classification.py`
4. Update the Pydantic models in `src/models/query_models.py`

## 📜 License

[MIT License]

## 📞 Contact

[abinesha312@gmail.com](mailto:abinesha312@gmail.com)
