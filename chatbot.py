import requests
import os
from dotenv import load_dotenv
load_dotenv()
def get_response(user_input):
    api_key=os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in environment variables")
    API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": f"Answer briefly as a fitness trainer: {user_input}"}],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers).json()
        print("API Response:", response)  
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error:", e)  
        return f"API Error: {str(e)}" 