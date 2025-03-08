import requests
import json
import re
from typing import List, Dict, Optional

class OllamaChatBot:
    def __init__(self, model: str, end_point_url: str, temperature=0.0, keep_history=True):
        self.model = model
        self.end_point_url = end_point_url
        self.temperature = temperature
        self.keep_history = keep_history
        self.chat_history: List[Dict[str, str]] = []

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
        # Add user message to history if keeping history
        if self.keep_history:
            self.chat_history.append({"role": "user", "content": prompt})
        
        # If we have chat history and keeping history, use the chat endpoint
        if self.keep_history and len(self.chat_history) > 1:
            payload = {
                "model": self.model,
                "messages": self.chat_history,
                "stream": False,
                "temperature": self.temperature
            }
            response = requests.post(f"{self.end_point_url}/api/chat", json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
            
            assistant_response = response.json().get("message", {}).get("content", "")
            
            # Add assistant response to history
            if self.keep_history:
                self.chat_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
        else:
            # For the first message or when not keeping history, use the generate endpoint
            payload = {
                "model": self.model, 
                "prompt": prompt, 
                "stream": False,
                "temperature": self.temperature
            }
            response = requests.post(f"{self.end_point_url}/api/generate", json=payload)

            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
                
            assistant_response = response.json().get("response", "")
            
            # Add assistant response to history if keeping history
            if self.keep_history:
                self.chat_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response

    def completeAsJSON(self, prompt: str) -> Optional[str]:
        # Modify the prompt to request JSON output if not already requesting it
        if "json" not in prompt.lower():
            json_prompt = f"{prompt}\n\nPlease respond in valid JSON format."
        else:
            json_prompt = prompt
        
        # Get response using the complete method
        response_text = self.complete(json_prompt)
        
        # Extract JSON from the response
        json_string = self.extract_markdown_content(response_text, "json")
        
        try:
            return json.dumps(json.loads(json_string), indent=2, ensure_ascii=False)
        except:
            print('Failed to parse LLM output: ', json_string)
            return None

    def clear_history(self) -> None:
        """
        Clear the chat history
        """
        self.chat_history = []
        
    def get_history(self) -> List[Dict[str, str]]:
        """
        Return the current chat history
        """
        return self.chat_history
        
    def set_keep_history(self, keep_history: bool) -> None:
        """
        Set whether to keep chat history
        """
        self.keep_history = keep_history

def main():
    # Example with history enabled (default)
    bot_with_history = OllamaChatBot(model="llama3.1:8b", end_point_url="http://localhost:11434")
    
    print("Example with history enabled (default):")
    response1 = bot_with_history.complete("Tell me a joke.")
    print("User: Tell me a joke.")
    print(f"Bot: {response1}\n")
    
    response2 = bot_with_history.complete("Tell me another one, but make it about programming.")
    print("User: Tell me another one, but make it about programming.")
    print(f"Bot: {response2}\n")
    
    print("History from bot_with_history:")
    for i, msg in enumerate(bot_with_history.get_history()):
        print(f"{i+1}. {msg['role']}: {msg['content']}")
    
    # Example with history disabled
    bot_without_history = OllamaChatBot(model="llama3.1:8b", end_point_url="http://localhost:11434", keep_history=False)
    
    print("\nExample with history disabled:")
    response3 = bot_without_history.complete("What is Python?")
    print("User: What is Python?")
    print(f"Bot: {response3}\n")
    
    response4 = bot_without_history.complete("What are its main features?")
    print("User: What are its main features?")
    print(f"Bot: {response4}\n")
    
    print("History from bot_without_history (should be empty or only last exchange):")
    for i, msg in enumerate(bot_without_history.get_history()):
        print(f"{i+1}. {msg['role']}: {msg['content']}")
    
    # Example of toggling history
    print("\nExample of toggling history:")
    bot_toggle = OllamaChatBot(model="llama3.1:8b", end_point_url="http://localhost:11434")
    
    response5 = bot_toggle.complete("Tell me about JavaScript.")
    print("User: Tell me about JavaScript.")
    print(f"Bot: {response5}\n")
    
    # Turn off history
    bot_toggle.set_keep_history(False)
    print("History keeping turned OFF")
    
    response6 = bot_toggle.complete("What about TypeScript?")
    print("User: What about TypeScript?")
    print(f"Bot: {response6}\n")
    
    # Turn history back on
    bot_toggle.set_keep_history(True)
    print("History keeping turned ON")
    
    response7 = bot_toggle.complete("How do they compare?")
    print("User: How do they compare?")
    print(f"Bot: {response7}\n")
    
    print("History from bot_toggle (should only have JavaScript and How do they compare?):")
    for i, msg in enumerate(bot_toggle.get_history()):
        print(f"{i+1}. {msg['role']}: {msg['content']}")


if __name__ == "__main__":
    main()