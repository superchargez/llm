from groq import Groq

client = Groq()
model = "llama3-8b-8192"

def chat_with_llm(messages):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=.9,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})
    response = chat_with_llm(messages)
    print(f"Assistant: {response}")
    messages.append({"role": "assistant", "content": response})