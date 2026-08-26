import re

class PromptProcessor:
    def __init__(self):
        self.max_length = 512
    
    def process(self, prompt: str) -> str:
        prompt = prompt.strip()
        prompt = re.sub(r'\s+', ' ', prompt)
        
        if len(prompt) > self.max_length:
            prompt = prompt[:self.max_length]
        
        return prompt
    
    def enhance(self, prompt: str) -> str:
        enhancements = [
            "high quality",
            "detailed",
            "professional"
        ]
        return f"{prompt}, {', '.join(enhancements)}"
