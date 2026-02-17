import torch
from diffusers import StableDiffusionPipeline
import io
import os
import uuid
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import motor.motor_asyncio
from datetime import datetime
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

app = FastAPI(title="TTI & TTS FastAPI Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cloudinary Configuration
cloudinary.config( 
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.getenv("CLOUDINARY_API_KEY"), 
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

# MongoDB Setup
MONGO_URI = os.getenv("MONGODB_URI")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.corenex_db
images_collection = db.generated_images
audio_collection = db.generated_audio

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Constants for local fallback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_ROOT = os.path.join(BASE_DIR, "media")
if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)

app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")

# --- Image Generation Service ---
class DiffusionService:
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiffusionService, cls).__new__(cls)
        return cls._instance

    def get_pipeline(self):
        if self._pipeline is None:
            model_id = "stabilityai/sd-turbo"
            try:
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
                    self._pipeline.enable_attention_slicing()
                
                print("SD-Turbo Pipeline loaded successfully.")
            except Exception as e:
                print(f"Error loading pipeline: {e}")
                return None
        return self._pipeline

    def generate(self, prompt, width=512, height=512):
        pipeline = self.get_pipeline()
        
        if pipeline is None:
            return None, self._generate_fallback(prompt, width, height)

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            steps = 4 if device == "cuda" else 1
            
            print(f"Generating image on {device} with SD-Turbo ({steps} step) for prompt: {prompt}")
            
            image = pipeline(prompt, width=width, height=height, num_inference_steps=steps, guidance_scale=0.0).images[0]
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join(MEDIA_ROOT, filename)
            image.save(filepath, format='PNG')
            
            return img_byte_arr, filename
        except Exception as e:
            print(f"Generation error: {e}")
            return None, self._generate_fallback(prompt, width, height)

    def _generate_fallback(self, prompt, width, height):
        img = Image.new('RGB', (width, height), color=(73, 109, 137))
        filename = f"fallback_{uuid.uuid4().hex}.png"
        filepath = os.path.join(MEDIA_ROOT, filename)
        img.save(filepath, format='PNG')
        return filename

service = DiffusionService()

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM" # Default Rachel voice

# --- Routes ---

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    img_data, local_filename = service.generate(request.prompt, request.width, request.height)
    
    upload_url = None
    if img_data:
        try:
            upload_result = cloudinary.uploader.upload(img_data, 
                folder = "tti_generated",
                public_id = local_filename.split('.')[0]
            )
            upload_url = upload_result.get("secure_url")
        except Exception as e:
            print(f"Cloudinary Upload Failed: {e}")
    
    final_url = upload_url if upload_url else f"/media/{local_filename}"
    
    image_doc = {
        "prompt": request.prompt,
        "image_url": final_url,
        "is_cloudinary": bool(upload_url),
        "created_at": datetime.utcnow(),
        "filename": local_filename
    }
    
    result = await images_collection.insert_one(image_doc)
    
    return {
        "status": "success",
        "id": str(result.inserted_id),
        "image_url": final_url,
        "prompt": request.prompt
    }

@app.post("/text-to-voice")
async def text_to_voice(request: TTSRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Reload key to ensure it's picked up from updated env
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key or api_key == "YOUR_ELEVENLABS_API_KEY":
        raise HTTPException(status_code=500, detail=f"ElevenLabs API Key not configured correctly. Found: {api_key[:5] if api_key else 'None'}")

    try:
        url = f"{ELEVENLABS_URL}/{request.voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": request.text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code != 200:
            error_detail = response.text
            if "missing_permissions" in error_detail:
                error_detail = "ElevenLabs API Key is missing 'text_to_speech' permissions. Please enable it in your ElevenLabs dashboard."
            raise HTTPException(status_code=response.status_code, detail=f"ElevenLabs API error: {error_detail}")

        audio_data = response.content
        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(MEDIA_ROOT, filename)
        
        # Save locally
        with open(filepath, "wb") as f:
            f.write(audio_data)

        # Upload to Cloudinary
        upload_url = None
        try:
            upload_result = cloudinary.uploader.upload(
                filepath, 
                resource_type="video", # Audio files use 'video' resource type in Cloudinary
                folder="tts_generated",
                public_id=filename.split('.')[0]
            )
            upload_url = upload_result.get("secure_url")
        except Exception as e:
            print(f"Cloudinary Audio Upload Failed: {e}")

        final_url = upload_url if upload_url else f"/media/{filename}"

        # Save to MongoDB
        audio_doc = {
            "text": request.text,
            "audio_url": final_url,
            "is_cloudinary": bool(upload_url),
            "created_at": datetime.utcnow(),
            "filename": filename
        }
        
        result = await audio_collection.insert_one(audio_doc)

        return {
            "status": "success",
            "id": str(result.inserted_id),
            "audio_url": final_url,
            "text": request.text
        }

    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images")
async def get_images():
    cursor = images_collection.find().sort("created_at", -1)
    images = await cursor.to_list(length=None)
    for img in images:
        img["id"] = str(img["_id"])
        del img["_id"]
    return images

@app.get("/voices")
async def get_voices():
    cursor = audio_collection.find().sort("created_at", -1)
    voices = await cursor.to_list(length=None)
    for v in voices:
        v["id"] = str(v["_id"])
        del v["_id"]
    return voices

@app.get("/")
async def root():
    return {"message": "TTI & TTS FastAPI is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
