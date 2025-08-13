TDS Data Analyst Agent
This project is a sophisticated, AI-powered data analyst agent. It uses a Large Language Model (LLM) within a Reason+Act (ReAct) framework to dynamically source, prepare, analyze, and visualize data based on natural language instructions.

The agent is designed to be flexible, resilient, and capable of handling a wide range of data analysis tasks, from web scraping to querying large datasets and generating visualizations.

🚀 Features
Agentic Framework: Built using a tool-based agentic architecture. The LLM can intelligently choose from a set of tools to accomplish a task.

Dynamic Code Generation: Generates and executes Python code on the fly to perform complex data manipulation, analysis, and visualization.

Secure Code Execution: Executes generated code in a secure environment to prevent malicious actions.

Toolbox:

Python REPL: For general-purpose data analysis using libraries like Pandas, NumPy, and Matplotlib.

Web Scraper: To fetch content from URLs.

FileSystem Tools: To list, read, and write files within its secure workspace.

Resilience & Self-Correction: The agent can reason about errors from its tools and attempt to correct its approach, making it robust.

Ready for Deployment: Includes configuration files (Dockerfile, render.yaml) for easy deployment to cloud platforms like Render.

⚙️ Project Structure
/data-analyst-agent
|
├── workspaces/         # Auto-created for temporary request files
|
├── app.py              # Main Flask application and API endpoint
├── requirements.txt    # Project dependencies
├── Dockerfile          # For building the deployment container
├── render.yaml         # Configuration for deploying to Render
├── .gitignore
├── LICENSE
|
└── agent/
    ├── __init__.py
    ├── agent.py        # Core ReAct agent logic
    ├── prompts.py      # System prompts for the LLM
    └── tools.py        # Definitions for the agent's tools

🛠️ Local Setup
Prerequisites
Python 3.9+

Pip

Git

Installation
Clone the repository:

git clone <your-repo-url>
cd data-analyst-agent

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Set up environment variables:
Create a file named .env in the project root and add your Google API key:

GENAI_API_KEY="your_google_ai_studio_api_key"

Run the application:

gunicorn app:app

The application will be running at http://127.0.0.1:8000.

☁️ Deployment to Render
This project is configured for easy deployment on Render.

Create a new public GitHub repository and push the project code to it.

Go to the Render Dashboard and create a New Web Service.

Connect your GitHub repository.

Render will automatically detect the render.yaml file. Give your service a name and click Create Web Service.

Render will then build and deploy your application. Your API will be live at the URL provided by Render.

🧪 How to Use the API
The API endpoint is /api/. It accepts POST requests with multipart/form-data.

You must provide a questions.txt file containing the instructions for the agent. You can also provide other data files (e.g., .csv, .png).

Example using cURL:
curl -X POST https://your-render-app-url/api/ \
     -F "questions.txt=@path/to/your/questions.txt" \
     -F "data.csv=@path/to/your/data.csv"

The agent will process the request and return a JSON response with the answers.
