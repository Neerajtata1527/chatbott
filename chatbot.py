# import requests
# def get_response(user_input):
#     API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "iySwJqYAAtCejNQrxNUq68ZdP0ehKfYO"
#     }
#     data = {
#         "model": "meta-llama/Meta-Llama-3-8B-Instruct",
#         "messages": [{"role": "user", "content": f"Answer briefly as a fitness trainer: {user_input}"}],
#         "max_tokens": 100
#     }
    
#     try:
#         response = requests.post(API_URL, json=data, headers=headers).json()
#         return response["choices"][0]["message"]["content"]
#     except:
#         return "Let's focus on fitness! Ask me about workouts or nutrition."
    
# import requests

# API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": "Bearer E1QD3KdkBWRZPDWvm45yZz7GgQ18p62c"  # Replace with your key
# }
# data = {
#     "model": "meta-llama/Meta-Llama-3-8B-Instruct",
#     "messages": [{"role": "user", "content": "How to gain muscle?"}],
#     "max_tokens": 100
# }

# response = requests.post(API_URL, json=data, headers=headers).json()
# print(response) 

import requests
def get_response(user_input):
    API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer E1QD3KdkBWRZPDWvm45yZz7GgQ18p62c"  # Double-check this!
    }
    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": f"Answer briefly as a fitness trainer: {user_input}"}],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers).json()
        print("API Response:", response)  # 👈 Debug line
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error:", e)  # 👈 Check Flask console for this
        return f"API Error: {str(e)}"  # Return the actual error