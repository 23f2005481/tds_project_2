import os
import subprocess
import requests
from bs4 import BeautifulSoup

class FileSystemTools:
    def __init__(self, workspace_dir: str):
        if not os.path.isdir(workspace_dir):
            raise ValueError(f"Workspace directory '{workspace_dir}' does not exist.")
        self.workspace_dir = os.path.abspath(workspace_dir)

    def _get_safe_path(self, file_path: str) -> str:
        safe_path = os.path.abspath(os.path.join(self.workspace_dir, file_path))
        if not safe_path.startswith(self.workspace_dir):
            raise ValueError("Error: Attempted to access a file outside the secure workspace.")
        return safe_path

    def list_files(self) -> str:
        try:
            files = os.listdir(self.workspace_dir)
            if not files:
                return "The workspace is empty."
            return "Files in workspace:\n" + "\n".join(files)
        except Exception as e:
            return f"Error listing files: {e}"

    def read_file(self, filename: str) -> str:
        try:
            safe_path = self._get_safe_path(filename)
            with open(safe_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file '{filename}': {e}"

    def write_file(self, filename: str, content: str) -> str:
        try:
            safe_path = self._get_safe_path(filename)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to file '{filename}'."
        except Exception as e:
            return f"Error writing to file '{filename}': {e}"


class WebScraperTool:
    def scrape(self, url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            return f"Error scraping URL '{url}': {e}"


class PythonREPLTool:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = os.path.abspath(workspace_dir)

    def run(self, code: str) -> str:
        try:
            process = subprocess.run(
                ['python', '-c', code],
                capture_output=True,
                text=True,
                cwd=self.workspace_dir,
                timeout=120
            )
            output = ""
            if process.stdout:
                output += f"STDOUT:\n{process.stdout}\n"
            if process.stderr:
                output += f"STDERR:\n{process.stderr}\n"
            
            if not output:
                return "Code executed successfully with no output."
                
            return output

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 120 seconds."
        except Exception as e:
            return f"Error executing Python code: {e}"

