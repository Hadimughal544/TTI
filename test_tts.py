import requests
import os
from dotenv import load_dotenv

# Load environment variables from TTI directory
load_dotenv(dotenv_path="c:/Users/786/OneDrive/Desktop/full stack FYP/TTI/.env")

def test_tts():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    api_url = "http://localhost:8002/text-to-voice"
    
    if not api_key or api_key == "YOUR_ELEVENLABS_API_KEY":
        print("❌ Error: ElevenLabs API Key not set in .env")
        return

    print(f"Testing TTS with API Key: {api_key[:5]}...{api_key[-5:]}")
    
    payload = {
        "text": "Hello, this is a test of the AI Studio text to voice system. I am working correctly.",
        "voice_id": "21m00Tcm4TlvDq8ikWAM"
    }
    
    try:
        response = requests.post(api_url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(f"Audio URL: {data.get('audio_url')}")
            print(f"Text: {data.get('text')}")
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure your uvicorn server is running on port 8002!")

if __name__ == "__main__":
    test_tts()
