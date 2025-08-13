import os
import uuid
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from agent.agent import run_agent

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WORKSPACE_DIR = os.path.join(os.getcwd(), "workspaces")
os.makedirs(WORKSPACE_DIR, exist_ok=True)


@app.route("/")
def index():
    return "Data Analyst Agent is running."


@app.route("/api/", methods=['POST'])
def api():
    request_id = str(uuid.uuid4())
    request_workspace = os.path.join(WORKSPACE_DIR, request_id)
    os.makedirs(request_workspace, exist_ok=True)
    logger.info(f"Created workspace: {request_workspace}")

    try:
        if 'questions.txt' not in request.files:
            logger.error("Bad Request: 'questions.txt' is missing.")
            return jsonify({"error": "questions.txt is a required file."}), 400

        uploaded_files = {}
        for filename, file_storage in request.files.items():
            file_path = os.path.join(request_workspace, filename)
            file_storage.save(file_path)
            uploaded_files[filename] = file_path
            logger.info(f"Saved file '{filename}' to '{file_path}'")

        with open(uploaded_files['questions.txt'], 'r') as f:
            prompt = f.read()

        logger.info(f"Invoking agent for request {request_id}...")
        result = run_agent(prompt, request_workspace)
        logger.info(f"Agent finished for request {request_id}.")

        return jsonify(result)

    except Exception as e:
        logger.exception(f"An unexpected error occurred for request {request_id}: {e}")
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

