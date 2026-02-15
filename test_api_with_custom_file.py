import requests

url = "http://localhost:8080/api/voice-detection"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "hackathon-secret"
}
# Read Base64 from file
with open("b64.txt", "r") as f:
    b64 = f.read().strip()

data = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": b64
}

response = requests.post(url, json=data, headers=headers)
print(response.json())  # Pretty JSON output
