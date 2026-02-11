import torch
from diffusers import StableDiffusionPipeline
import io
import os
from PIL import Image
from django.conf import settings
from django.core.files.base import ContentFile

class DiffusionService:
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiffusionService, cls).__new__(cls)
        return cls._instance

    def get_pipeline(self):
        if self._pipeline is None:
            # Using SD-Turbo for significantly faster generation on CPU
            model_id = "stabilityai/sd-turbo"
            try:
                # Use CPU by default if GPU is not available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                
                print(f"Loading SD-Turbo Pipeline on {device}...")
                self._pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
                self._pipeline.to(device)
                if device == "cpu":
                    # Optimization for CPU and limited RAM
                    self._pipeline.enable_attention_slicing()
                
                print("SD-Turbo Pipeline loaded successfully.")
            except Exception as e:
                print(f"Error loading pipeline: {e}")
                return None
        return self._pipeline

    def generate(self, prompt, width=512, height=512):
        pipeline = self.get_pipeline()
        
        if pipeline is None:
            print("Using fallback image generation (mock)...")
            return self._generate_fallback(prompt, width, height)

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # SD-Turbo is designed for 1-4 steps. 
            # 1 step is extremely fast for CPU.
            steps = 4 if device == "cuda" else 1
            
            print(f"Generating image on {device} with SD-Turbo ({steps} step) for prompt: {prompt}")
            
            # Run the generation
            # For SD-Turbo, guidance_scale=0.0 is often recommended for 1-step
            image = pipeline(prompt, width=width, height=height, num_inference_steps=steps, guidance_scale=0.0).images[0]
            
            # Save to buffer
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            print("Image generated successfully.")
            return buf.getvalue()
        except Exception as e:
            print(f"Generation error: {e}")
            return self._generate_fallback(prompt, width, height)

    def _generate_fallback(self, prompt, width, height):
        # Create a placeholder image with the prompt text
        img = Image.new('RGB', (width, height), color=(73, 109, 137))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.text((width//10, height//2), f"Generating/Timed Out: {prompt[:30]}...", fill=(255, 255, 0))
        d.text((width//10, height//2 + 20), "(Check console for progress)", fill=(200, 200, 200))
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

diffusion_service = DiffusionService()
