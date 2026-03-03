from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from utils.pdf_utils import extract_text_from_pdf
from services.llm_service import get_llm_answer, stream_llm_answer
from models.request_models import ChatRequest, ChatResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
origins = os.getenv("CORS_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resume once at startup
resume_text = extract_text_from_pdf("resume.pdf")

# ==========================
# 🔹 SYSTEM CONSTANTS
# ==========================

CHATBOT_EXPLANATION = (
    "Ardhendu built this AI chatbot to demonstrate practical implementation "
    "of Large Language Models using FastAPI and a vectorless RAG architecture. "
    "It showcases his ability to integrate AI into real-world applications "
    "and create an interactive, recruiter-friendly portfolio experience."
)

CONTACT_RESPONSE = "Please use the contact form on the website to get in touch."

# ==========================
# 🔹 INTENT DETECTION LAYER
# ==========================

def is_chatbot_intent(question: str) -> bool:
    question = question.lower()
    return (
        "chatbot" in question and
        any(word in question for word in ["why", "purpose", "build", "create"])
    )

def is_contact_intent(question: str) -> bool:
    question = question.lower()
    return any(word in question for word in [
        "email",
        "phone",
        "contact number",
        "mobile",
        "whatsapp"
    ])

# ==========================
# 🔹 PROMPT BUILDER
# ==========================

def build_prompt(user_question: str, resume_text: str) -> str:
    return f"""
You are Ardhendu's professional AI representative and personal PR assistant.
You represent him on his portfolio website.

Your goals:
- Present Ardhendu as skilled, competent, and professional.
- Highlight strengths clearly and confidently.
- Keep responses concise but impactful.
- Maintain a confident, polished, recruiter-friendly tone.

Rules:
- Only use information explicitly provided in the resume text.
- Do NOT invent, assume, or infer anything not clearly stated.
- Do NOT reveal phone number, email, or personal contact details.
- If asked for personal contact information, respond:
  "{CONTACT_RESPONSE}"
- If the question is unrelated to Ardhendu’s professional background, politely decline.
- If the answer is not clearly stated in the resume, respond:
  "That specific detail is not mentioned in the resume, but you may contact Ardhendu for more information."

Resume Information:
{resume_text}

User Question:
{user_question}

Provide a confident, professional response based only on the resume above.
"""

# ==========================
# 🔹 CHAT ENDPOINTS
# ==========================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()
    # Handle chatbot-purpose questions directly
    if is_chatbot_intent(question):
        return ChatResponse(answer=CHATBOT_EXPLANATION)
    # Handle contact info questions directly
    if is_contact_intent(question):
        return ChatResponse(answer=CONTACT_RESPONSE)
    # Resume-based RAG
    prompt = build_prompt(question, resume_text)
    answer = await get_llm_answer(prompt)
    return ChatResponse(answer=answer)

# Streaming endpoint
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    question = request.question.strip()
    # Handle chatbot-purpose questions directly
    if is_chatbot_intent(question):
        def fake_stream():
            for word in CHATBOT_EXPLANATION.split():
                yield word + " "
        return StreamingResponse(fake_stream(), media_type="text/plain")
    # Handle contact info questions directly
    if is_contact_intent(question):
        def fake_stream():
            for word in CONTACT_RESPONSE.split():
                yield word + " "
        return StreamingResponse(fake_stream(), media_type="text/plain")
    # Resume-based RAG
    prompt = build_prompt(question, resume_text)
    return StreamingResponse(stream_llm_answer(prompt), media_type="text/plain")