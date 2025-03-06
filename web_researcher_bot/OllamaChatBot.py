import requests
import json
import re

class OllamaChatBot:
    def __init__(self, model: str, end_point_url: str, temperature=0.0):
        self.model = model
        self.end_point_url = end_point_url
        self.temperature = temperature


    def extract_markdown_content(self, text: str, type: str = "json") -> str:
        start = f"""```{type}"""
        end = """```"""

        start_idx = text.find(start)
        end_idx = text.rfind(end)

        if start_idx >= 0 and end_idx >= 0:
            start_idx += len(type) + 3
            end_idx -= 1
            return (text[start_idx:end_idx]).strip()

        return text.strip()

    def complete(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        response = requests.post(f"{self.end_point_url}/api/generate", json=payload)

        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def completeAsJSON(self, prompt: str):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
        }
        response = requests.post(f"{self.end_point_url}/api/generate", json=payload)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        # The API returns a JSON object with a 'response' field containing the JSON string
        json_string = response.json().get("response", "{}")
        json_string = self.extract_markdown_content(json_string, "json")

        try:
            return json.dumps(json.loads(json_string), indent=2, ensure_ascii=False)
        except:
            print('Failed to parse LLM output: ', json_string)
            return None

def main():
    # Note: This implementation does not keep an on-going chat (it's a one-shot prompt interaction)
    bot = OllamaChatBot(model="llama3.1:8b", end_point_url="http://localhost:11434")
    prompt = "Tell me a joke."

    # Using complete() method
    response_text = bot.complete(prompt)
    print("Using complete() method:")
    print(f"Prompt: {prompt}")
    print(f"Response: {response_text}\n")

    # Using completeJSON() method
    response_json = bot.completeAsJSON(prompt)
    print("Using completeJSON() method:")
    print(f"Prompt: {prompt}")
    print(f"Response: {response_json}")


if __name__ == "__main__":
    main()
