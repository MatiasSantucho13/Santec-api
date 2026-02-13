import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

system_instruction = """
Sos el asistente virtual de SanTec Software. 
Objetivo: Vender soluciones de IA y Web a Inmobiliarias y Constructoras.
Personalidad: Profesional, breve, persuasivo.
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# --- MODELO DE DATOS ACTUALIZADO ---
# Ahora aceptamos "history" (la memoria)
class ChatMessage(BaseModel):
    message: str
    history: List[Dict[str, Any]] = [] 

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    if not api_key:
        return {"response": "Error: Falta API KEY."}
    
    try:
        # 1. Iniciamos el chat CON LA HISTORIA que nos manda el usuario
        # (Así el bot recuerda lo que hablaron antes)
        chat = model.start_chat(history=request.history)
        
        # 2. Enviamos el mensaje nuevo
        response = chat.send_message(request.message)
        
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Hubo un error: {str(e)}"}
