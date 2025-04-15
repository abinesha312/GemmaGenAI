# UNT AI Assistant

A high-performance multi-agent AI assistant designed for the University of North Texas. Built on the **Gemma 3 27B** model and accelerated with **vLLM**, the system provides targeted academic support via a web-based interface powered by **Chainlit**.

---

## 🎥 Demo

### Video Walkthrough

[![Application Demo]](https://youtu.be/zfd0Z0ckva0)

---

## 🖼 Interface

### Real-Time System View

![Application Screenshot](image.png)

_Screenshot shows eight NVIDIA H100 GPUs running parallel inference threads with load distributions across GPUs 4–7 using `vLLM`._

---

## 🔍 Features

- **Specialized Agents** for academic scenarios:

  - Email Composition
  - Research Paper Support
  - Academic Concepts Guide
  - UNT Resources Navigator
  - General Campus Information

- **Modular Multi-Agent System**:

  - Intelligent query classification with TF-IDF and cosine similarity
  - Clean, structured response formatting

- **Optimized Inference Pipeline**:

  - Runs on NVIDIA H100 with fine-tuned batching via `vLLM`
  - Efficient memory and compute utilization across GPUs

- **Interactive UI**:
  - Built with Chainlit for smooth real-time interactions
  - Custom routing for each academic use case

---

## 📊 Benchmarks (Observed)

| Metric                         | Value                               |
| ------------------------------ | ----------------------------------- |
| GPUs Used                      | 8x NVIDIA H100                      |
| Inference Engine               | vLLM + Gemma 3 27B                  |
| Max GPU Utilization (Observed) | ~53% on GPU 4 and GPU 5             |
| GPU Memory Allocation          | ~50.6–59.6 GB across active GPUs    |
| Power Draw                     | ~91W per GPU (under load)           |
| Input Model Length             | 4096 tokens                         |
| Tensor Parallelism             | 4                                   |
| Launch Mode                    | Multi-GPU, Multi-process via Podman |

> _Note: GPUs 0–3 remain idle, while GPUs 4–7 actively handle batched inference requests from concurrent clients._

---

## 🧱 Architecture

- **vLLM Inference Server**:

  - Hosts the Gemma-3-27B model
  - Optimized for parallel generation

- **Chainlit Frontend**:
  - Manages UI and routes queries to appropriate agents
  - Provides response formatting and interface customization

---

## 🛠 Installation

### Prerequisites

- Podman
- Python 3.10+
- NVIDIA GPU with CUDA support (tested with H100)

### Setup Steps

```bash
git clone https://github.com/yourusername/unt-ai-assistant.git
cd unt-ai-assistant

mkdir -p models/FAISS_INGEST/vectorstore
mkdir -p logs

chmod +x podman_run.sh
./podman_run.sh
```
