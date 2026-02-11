import torch
from diffusers import StableDiffusionPipeline
import os

def test_gen():
    print("--- Local Generation Test ---")
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cpu" # We know CUDA is False
    torch_dtype = torch.float32
    
    try:
        print(f"Loading model '{model_id}' on {device}...")
        print("Note: This will attempt to download ~5GB if not already cached.")
        # Try to load with local_files_only first to see if it's there
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, local_files_only=True)
            print("Model loaded from local cache.")
        except Exception:
            print("Model not in cache. Attempting download (this requires internet and time)...")
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            
        pipe.to(device)
        print("Model ready. Generating...")
        
        prompt = "a raining day and a lots of dogs are barking"
        image = pipe(prompt, num_inference_steps=5).images[0] # Low steps for speed test
        
        image.save("test_gen_result.png")
        print("Success! Image saved as test_gen_result.png")
        
    except Exception as e:
        print(f"FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gen()
