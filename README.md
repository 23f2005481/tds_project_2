
# **TDS Data Analyst Agent**

> 🧠 Autonomous AI-powered Data Analyst — source, prepare, analyze & visualize data from natural language instructions.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Framework](https://img.shields.io/badge/Framework-Flask-orange?logo=flask)

---

## 🚀 Overview

The **TDS Data Analyst Agent** is a **Reason + Act (ReAct)** AI framework powered by an LLM that can dynamically perform **data sourcing, cleaning, analysis, and visualization**.
It intelligently chooses tools, generates Python code, and executes it securely — handling everything from **web scraping** to **large dataset queries** and **custom charts**.

---

## ✨ Features

* 🤖 **Agentic Framework** – LLM selects and uses tools based on tasks.
* ⚡ **Dynamic Code Execution** – Generates Python code for Pandas, NumPy, Matplotlib, etc.
* 🔒 **Secure Sandbox** – Isolated environment prevents malicious actions.
* 🛠 **Toolbox**:

  * Python REPL for data analysis
  * Web Scraper for real-time data fetching
  * File system tools for secure file handling
* 🔄 **Resilience** – Detects errors, retries with self-corrections.
* ☁ **Deployment Ready** – Pre-configured with Docker & Render YAML.

---

## 📂 Project Structure

```
data-analyst-agent/
│
├── workspaces/         # Temp storage for each request
│
├── app.py              # Flask API application
├── requirements.txt    # Dependencies
├── Dockerfile          # Container build config
├── render.yaml         # Render deployment config
├── .gitignore
├── LICENSE
│
└── agent/
    ├── __init__.py
    ├── agent.py        # ReAct agent logic
    ├── prompts.py      # System prompts for LLM
    └── tools.py        # Tool definitions
```

---

## ⚙️ Setup

<details>
<summary><strong>📌 Local Installation</strong></summary>

**Prerequisites**

* Python 3.9+
* Pip
* Git

**Steps**

```bash
# Clone repository
git clone <your-repo-url>
cd data-analyst-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo 'GENAI_API_KEY="your_google_ai_studio_api_key"' > .env

# Run the application
gunicorn app:app
```

App runs on: `http://127.0.0.1:8000`

</details>

---

## ☁ Deployment to Render

<details>
<summary><strong>🚀 Render Deployment Guide</strong></summary>

1. Push code to a **public GitHub repo**.
2. On **Render Dashboard** → Click **New Web Service**.
3. Connect your repo.
4. Render detects `render.yaml` → Configure service name → Deploy.
5. Your API will be live at the provided Render URL.

</details>

---

## 🧪 API Usage

**Endpoint:** `/api/`
**Method:** `POST`
**Type:** `multipart/form-data`

**Required:**

* `questions.txt` → Contains natural language instructions for the agent.

**Optional:**

* Additional data files (CSV, PNG, etc.) for analysis.

**Example Request:**

```bash
curl -X POST https://your-render-app-url/api/ \
     -F "questions.txt=@path/to/questions.txt" \
     -F "data.csv=@path/to/data.csv"
```

**Example Response:**

```json
{
  "status": "success",
  "answers": [
    "Data summary generated...",
    "Visualization saved as chart.png"
  ]
}
```

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---
