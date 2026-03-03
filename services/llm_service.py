import os
import httpx
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

async def get_llm_answer(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GROQ_API_URL, headers=headers, json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"]

# Streaming version
async def stream_llm_answer(prompt: str, model: str = "llama-3.1-8b-instant"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    print("Starting to stream response from LLM...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", GROQ_API_URL, headers=headers, json=payload) as response:      #Open a streaming HTTP connection:
            async for line in response.aiter_lines():  #Read the response line by line:
                if line.strip():
                    # Groq uses SSE format: "data: {json}" The LLM responds using Server-Sent Events (SSE), where each line starts with data: and contains a JSON chunk of the answer.
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str.strip() == "[DONE]":  #Handle end of stream
                            print("Stream complete")
                            break
                        try:
                            data = json.loads(json_str)
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                print(f"Streaming chunk: {repr(content)}")
                                yield content   #sends the chunk to FastAPI, which streams it to the frontend.
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}, line: {line}")
                            continue
                        except Exception as e:
                            print(f"Error processing chunk: {e}")
                            continue