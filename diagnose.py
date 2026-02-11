import sys
import os

def check_env():
    print("--- Environment Check ---")
    print(f"Python Version: {sys.version}")
    
    try:
        import torch
        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("Error: torch is NOT installed.")

    try:
        import diffusers
        print(f"Diffusers version: {diffusers.__version__}")
    except ImportError:
        print("Error: diffusers is NOT installed.")

    try:
        from PIL import Image
        print("PIL is installed.")
    except ImportError:
        print("Error: Pillow is NOT installed.")

    print("\n--- Model Path Check ---")
    # Check if model might be downloaded (default cache)
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        print(f"HuggingFace cache exists at: {cache_dir}")
        print(f"Files found in cache: {len(os.listdir(cache_dir))}")
    else:
        print("HuggingFace cache not found (Model might not be downloaded yet).")

if __name__ == "__main__":
    check_env()
