# GenAI Systems: UNT Intelligent Assistant

Welcome to **GenAI Systems**, your intelligent assistant tailored for the **University of North Texas (UNT)** community. Whether you're a prospective student, current student, faculty member, or visitor, GenAI Systems is here to assist with all UNT-related queries.

---

## 🌟 Features

GenAI Systems can help you with:

- **🎓 Admissions & Programs**: Get detailed information about UNT’s academic programs, admission requirements, tuition fees, and scholarships.
- **🏛️ Campus Life**: Discover campus resources, events, student services, and organizations.
- **📜 Policies & Deadlines**: Stay updated on UNT’s academic guidelines, deadlines, and policies.
- **🔬 Research Opportunities**: Explore UNT’s research initiatives, faculty expertise, and student organizations.
- **🛠 Technical & Administrative Support**: Resolve UNT-related technical or administrative issues efficiently.

---

## ⚙️ How It Works

1. **Ask a Question**: Simply type your UNT-related query into the chat interface.
2. **Receive Answers**: GenAI Systems provides accurate, structured responses tailored to your needs.
3. **Specialized Agents**:
   - ✉️ **Email Composer**: Helps draft professional emails for academic settings.
   - 📖 **Research Paper Assistant**: Guides students in structuring and developing research papers.
   - 🧠 **Academic Concepts Guide**: Explains complex academic theories and concepts.
   - 🔗 **Resource Redirector**: Directs users to relevant UNT resources and websites.
   - 🏫 **General UNT Assistant**: Provides general UNT-related information.

---

## 🛠 Installation Guide

To deploy GenAI Systems locally, follow these steps:

### 1️⃣ Clone the Repository

```sh
git clone <repository-url>
cd <repository-directory>
```

### 2️⃣ Build Docker Images

Build the application container:

```sh
docker build -t genai-systems .
```

Build the vLLM server container:

```sh
docker build -f Dockerfile.vllm -t vllm-server .
```

### 3️⃣ Run Containers

Start the vLLM server:

```sh
docker run --name vllm-server -p 5000:5000 vllm-server
```

Start the Chainlit application:

```sh
docker run --name genai-systems -p 8000:8000 genai-systems
```

---

## 🔧 Configuration

### 🌍 Environment Variables

Set the following environment variables in `.env` or `Dockerfile`:

```ini
MODEL_ID=google/gemma-3-27b-it
INFERENCE_SERVER_URL=http://vllm-server:5000/v1
MAX_RETRIES=3
RETRY_DELAY=2  # in seconds
REQUEST_TIMEOUT=30  # in seconds
```

### ⚙️ Application Settings (`Configure.toml`)

- **Telemetry**: Enable or disable (`enable_telemetry = true`).
- **Session Timeout**: Set session expiration (`user_session_timeout = 1296000` seconds).
- **CORS Configuration**: Define allowed origins (`allow_origins = ["*"]`).

---

## 📦 Dependencies

Install Python dependencies listed in `requirements.txt`:

```sh
pip install -r requirements.txt
```

Key dependencies:

- `chainlit>=0.7.0`
- `openai>=1.0.0`
- `python-dotenv>=1.0.0`
- `requests>=2.31.0`
- `tenacity>=8.2.0`

---

## 🚀 Usage

Visit the application at [http://genai.unt.edu](http://genai.unt.edu) after starting the containers.

🔐 **Access Note**: The system is behind a UNT proxy server. Request access from the university if needed.

### 🔹 Starter Prompts

Use predefined prompts for common tasks:

- "Compose an email to a professor."
- "Structure a research paper."
- "Explain academic concepts like quantum mechanics."
- "Redirect me to UNT graduate admissions requirements."

---

## 🤝 Contributing

We welcome contributions! Follow these steps:

1. **Fork** the repository.
2. **Create a branch**: `git checkout -b feature-name`
3. **Commit changes**: `git commit -m "Add feature"`
4. **Push branch**: `git push origin feature-name`
5. **Submit a pull request** 🎉

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📩 Contact

For support or inquiries, reach out at [abinesha312@gmail.com](mailto:abinesha312@gmail.com).
