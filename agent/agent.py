import os
import json
import logging
import google.generativeai as genai

from .prompts import SYSTEM_PROMPT
from .tools import FileSystemTools, WebScraperTool, PythonREPLTool

logger = logging.getLogger(__name__)

if "GENAI_API_KEY" not in os.environ:
    raise ValueError("GENAI_API_KEY environment variable not set.")
genai.configure(api_key=os.environ["GENAI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash-latest')


def run_agent(prompt: str, workspace_dir: str) -> dict:
    fs_tools = FileSystemTools(workspace_dir)
    web_scraper = WebScraperTool()
    python_repl = PythonREPLTool(workspace_dir)

    tool_map = {
        "FileSystemTools.list_files": fs_tools.list_files,
        "FileSystemTools.read_file": fs_tools.read_file,
        "FileSystemTools.write_file": fs_tools.write_file,
        "WebScraperTool.scrape": web_scraper.scrape,
        "PythonREPLTool.run": python_repl.run,
    }

    conversation_history = [
        SYSTEM_PROMPT,
        f"User Request: {prompt}"
    ]
    
    max_iterations = 15
    for i in range(max_iterations):
        logger.info(f"Agent Iteration {i+1}/{max_iterations}")

        llm_response_str = None
        for attempt in range(3):
            try:
                response = model.generate_content("\n".join(conversation_history))
                llm_response_str = response.text
                break
            except Exception as e:
                logger.warning(f"LLM generation failed (attempt {attempt+1}): {e}")
                if attempt == 2:
                    raise Exception("LLM failed to respond after multiple attempts.") from e
        
        try:
            action = json.loads(llm_response_str)
            thought = action.get("thought", "(No thought provided)")
            tool_call = action.get("tool_call")
            logger.info(f"Agent Thought: {thought}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode LLM response into JSON: {llm_response_str}")
            conversation_history.append(f"Observation: Your last response was not valid JSON. Please respond with a valid JSON object containing 'thought' and 'tool_call'.")
            continue

        conversation_history.append(llm_response_str)

        if tool_call is None:
            logger.info("Agent decided to finish.")
            break

        tool_name = tool_call.get("tool_name")
        parameters = tool_call.get("parameters", {})
        
        if tool_name not in tool_map:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                tool_function = tool_map[tool_name]
                observation = tool_function(**parameters)
                logger.info(f"Executed tool '{tool_name}' with params {parameters}. Observation: {observation[:200]}...")
            except Exception as e:
                observation = f"Error executing tool '{tool_name}': {e}"
                logger.error(observation)
        
        conversation_history.append(f"Observation: {observation}")

    result_path = os.path.join(workspace_dir, "result.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError("Agent finished but the required 'result.json' was not created.")

    with open(result_path, 'r') as f:
        final_result = json.load(f)

    return final_result
