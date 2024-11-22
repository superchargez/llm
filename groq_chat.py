import requests

url = "http://127.0.0.1:8000/chat"
messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})
    response = requests.post(url, json={"messages": messages})
    assistant_response = response.json()["response"]
    print(f"Assistant: {assistant_response}")
    messages.append({"role": "assistant", "content": assistant_response})