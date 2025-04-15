# GenAI Systems: UNT Intelligent Assistant

Welcome to **GenAI Systems**, your go-to intelligent assistant designed to provide comprehensive support for the University of North Texas (UNT) community. Whether you're a prospective student, current student, faculty member, or visitor, GenAI Systems is here to assist with all UNT-related queries.

---

## Features

GenAI Systems can help you with:

- **Admissions and Programs**: Get detailed information about UNT’s academic programs, admissions requirements, tuition fees, and scholarships.
- **Campus Life**: Learn about campus resources, events, and student services.
- **Policies and Deadlines**: Stay informed about UNT’s policies, academic guidelines, and important deadlines.
- **Research Opportunities**: Discover UNT’s research initiatives, faculty expertise, and student organizations.
- **Technical and Administrative Support**: Solve UNT-related technical issues or administrative queries.

---

## How It Works

1. **Ask a Question**: Simply type your UNT-related query into the chat interface.
2. **Receive Answers**: GenAI Systems provides accurate and structured responses tailored to your needs.
3. **Specialized Agents**:
   - _Email Composer_: Helps draft professional emails for academic settings.
   - _Research Paper Assistant_: Guides students in structuring and developing research papers.
   - _Academic Concepts Guide_: Explains complex academic concepts and theories.
   - _Resource Redirector_: Directs users to relevant UNT resources and websites.
   - _General UNT Assistant_: Provides general information about UNT.

---

## Installation

To deploy GenAI Systems locally, follow these steps:

### 1. Clone the Repository

git clone <repository-url>
cd <repository-directory>

text

### 2. Build Docker Images

Build the application container:
docker build -t genai-systems .

text

Build the vLLM server container:
docker build -f Dockerfile.vllm -t vllm-server .

text

### 3. Run Containers

Start the vLLM server:
docker run --name vllm-server -p 5000:5000 vllm-server

text

Start the Chainlit application:
docker run --name genai-systems -p 8000:8000 genai-systems

text

---

## Configuration

### Environment Variables

Set environment variables in `Dockerfile` or `.env` file:

- `MODEL_ID`: AI model identifier (default: `google/gemma-3-27b-it`)
- `INFERENCE_SERVER_URL`: URL for the inference server (default: `http://vllm-server:5000/v1`)
- `MAX_RETRIES`: Number of retries for API calls (default: `3`)
- `RETRY_DELAY`: Delay between retries (default: `2` seconds)
- `REQUEST_TIMEOUT`: Timeout for API requests (default: `30` seconds)

### Application Settings

Modify settings in `Configure.toml`:

- **Telemetry**: Enable or disable telemetry (`enable_telemetry = true`).
- **Session Timeout**: Set session expiration duration (`user_session_timeout = 1296000` seconds).
- **CORS Configuration**: Define allowed origins (`allow_origins = ["*"]`).

---

## Dependencies

Install Python dependencies listed in `requirements.txt`:

- `chainlit>=0.7.0`
- `openai>=1.0.0`
- `python-dotenv>=1.0.0`
- `requests>=2.31.0`
- `tenacity>=8.2.0`

---

## Usage

Visit the application at [http://genai.unt.edu](http://genai.unt.edu) after starting the containers.

### Starter Prompts

Use predefined starter prompts for common tasks:

- Compose an email to a professor.
- Structure a research paper.
- Explain academic concepts like quantum mechanics.
- Redirect to UNT graduate admissions requirements.

---

## Contributing

We welcome contributions from the community! To contribute:

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push your branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License.

---

## Contact

For support or inquiries, please contact us at [abinesha312@gmail.com.com](mailto:abinesha312@gmail.com).
