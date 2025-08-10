from flask import Flask, request, jsonify
from analysis import handle_analysis
import tempfile
import os

app = Flask(__name__)

@app.route("/api/", methods=["POST"])
def api():
    try:
        if 'questions.txt' not in request.files:
            return jsonify({"error": "questions.txt is required"}), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            files = {}
            for f in request.files:
                filepath = os.path.join(tmpdir, f)
                request.files[f].save(filepath)
                files[f] = filepath

            result = handle_analysis(files)
            return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200
