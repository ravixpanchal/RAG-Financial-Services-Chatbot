import os
import requests
import json
import sys

# Read the system prompt we created earlier
prompt_path = "./financial-services/plugins/agent-plugins/fin-query-optimizer/agents/fin-query-optimizer.md"
skills_dir = "./financial-services/plugins/agent-plugins/fin-query-optimizer/skills"

try:
    with open(prompt_path, "r") as f:
        # Extract the prompt part, removing the markdown frontmatter
        content = f.read()
        system_prompt = content.split("---")[-1].strip()
except FileNotFoundError:
    print(f"Could not find the system prompt at {prompt_path}")
    exit(1)

# Dynamically load skills from the plugin directory if present
if os.path.exists(skills_dir) and os.path.isdir(skills_dir):
    skills_content = []
    for root, dirs, files in os.walk(skills_dir):
        for file in files:
            if file.lower() == "skill.md":
                skill_path = os.path.join(root, file)
                try:
                    with open(skill_path, "r") as f:
                        skill_data = f.read()
                        # Extract skill content, removing the frontmatter
                        skill_body = skill_data.split("---")[-1].strip()
                        skill_name = os.path.basename(os.path.dirname(skill_path))
                        skills_content.append(f"### Skill: {skill_name}\n{skill_body}")
                except Exception as e:
                    print(f"Warning: Failed to load skill at {skill_path}: {e}")
    if skills_content:
        system_prompt += "\n\n## Bundled Skills\n\n" + "\n\n".join(skills_content)

def optimize_query(user_query: str):
    """
    Sends the user query to local Ollama API using the FinQueryOptimizer system prompt.
    """
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": "qwen2.5:7b-instruct", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "options": {
            "temperature": 0.0 # Low temperature for deterministic output
        },
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Make sure Ollama is running locally on port 11434.")
        return

    if response.status_code == 200:
        result = response.json()
        try:
            # Extract and print the raw text response
            content_str = result['message']['content']
            print(content_str.strip())
            return content_str
        except KeyError as e:
            print("Error extracting content from response:")
            print(result)
            print(f"Exception: {e}")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
        optimize_query(test_query)
    else:
        print("FinQueryOptimizer Interactive Mode (type 'exit' or 'quit' to stop)")
        while True:
            try:
                test_query = input("\nEnter your query: ")
                if test_query.strip().lower() in ['exit', 'quit']:
                    break
                if not test_query.strip():
                    continue
                optimize_query(test_query)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
