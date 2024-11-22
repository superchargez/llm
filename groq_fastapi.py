from fastapi import FastAPI, Request
from pydantic import BaseModel
from groq import Groq

app = FastAPI()
client = Groq()
model = "llama-3.1-70b-versatile"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.post("/chat")
async def chat(request: ChatRequest):
    completion = client.chat.completions.create(
        model=model,
        messages=[message.dict() for message in request.messages],
        temperature=1,
        max_tokens=1024,
        top_p=.9,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return {"response": response}