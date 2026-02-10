import torch
from diffusers import StableDiffusionPipeline
from django.conf import settings
from django.core.files.base import ContentFile
import io
import os
from PIL import Image

class DiffusionService:
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiffusionService, cls).__new__(cls)
        return cls._instance

    def get_pipeline(self):
        if self._pipeline is None:
            model_id = "runwayml/stable-diffusion-v1-5"
            try:
                # Use CPU by default if GPU is not available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                
                print(f"Loading Diffusion Pipeline on {device}...")
                self._pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
                self._pipeline.to(device)
                print("Pipeline loaded successfully.")
            except Exception as e:
                print(f"Error loading pipeline: {e}")
                return None
        return self._pipeline

    def generate(self, prompt, width=512, height=512):
        pipeline = self.get_pipeline()
        
        if pipeline is None:
            # Fallback for development/environments without the model
            print("Using fallback image generation (mock)...")
            return self._generate_fallback(prompt, width, height)

        try:
            # Run the generation
            image = pipeline(prompt, width=width, height=height).images[0]
            
            # Save to buffer
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            return buf.getvalue()
        except Exception as e:
            print(f"Generation error: {e}")
            return self._generate_fallback(prompt, width, height)

    def _generate_fallback(self, prompt, width, height):
        # Create a placeholder image with the prompt text
        img = Image.new('RGB', (width, height), color=(73, 109, 137))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.text((width//4, height//2), f"Prompt: {prompt[:20]}...", fill=(255, 255, 0))
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

diffusion_service = DiffusionService()
