import requests

rasa_url = "http://localhost:5005/webhooks/rest/webhook"
url = "http://localhost:5005/model/parse"

user_message = "I want machine learning teacher?"

payload = {
    "sender": "user",
    "message": user_message
}

response = requests.post(rasa_url, json=payload)

response_json = response.json()

data = {
    "text": user_message
}

response = requests.post(url, json=data)

print(response.text) 
