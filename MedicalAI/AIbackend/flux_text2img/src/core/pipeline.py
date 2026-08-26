from pathlib import Path
from .engine import FluxEngine
from ..utils.image_utils import save_image
from ..preprocessing.prompt_processor import PromptProcessor

class FluxPipeline:
    def __init__(self):
        self.engine = FluxEngine()
        self.processor = PromptProcessor()
    
    def __call__(self, prompt: str, output_path: str = None, **kwargs):
        processed_prompt = self.processor.process(prompt)
        image_bytes = self.engine.generate(processed_prompt, **kwargs)
        
        if output_path:
            save_image(image_bytes, output_path)
        
        return image_bytes
