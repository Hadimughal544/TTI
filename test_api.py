import requests
import json
import time

def test_api():
    api_key = "ipqI2LE1yfNh8hVdbeyiadaWuOu35SUrJJ4LcAn7ajQiU1ZNFCdbRINSfimb"
    url = "https://modelslab.com/api/v7/images/text-to-image"
    
    payload = json.dumps({
        "key": api_key,
        "prompt": "A beautiful landscape",
        "negative_prompt": "bad quality",
        "width": "512",
        "height": "512",
        "samples": "1",
        "num_inference_steps": "20",
        "safety_checker": "yes"
    })

    headers = {
        'Content-Type': 'application/json'
    }

    print(f"Testing ModelsLab API...")
    try:
        response = requests.post(url, headers=headers, data=payload)
        print(f"Status Code: {response.status_code}")
        response_data = response.json()
        print(f"Response Object: {json.dumps(response_data, indent=2)}")
        
        if response_data.get('status') == 'processing':
            eta = response_data.get('eta', 10)
            print(f"Image is processing. ETA: {eta}s. Waiting...")
            fetch_url = response_data.get('fetch_result')
            if fetch_url:
                time.sleep(eta + 5)
                print(f"Fetching result from: {fetch_url}")
                fetch_payload = json.dumps({"key": api_key})
                fetch_res = requests.post(fetch_url, headers=headers, data=fetch_payload)
                print(f"Fetch Response: {fetch_res.json()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
