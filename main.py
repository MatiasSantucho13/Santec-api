import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURACIÓN ---
app = FastAPI()

# Permisos para que tu web (y tu compu) puedan usar el bot
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://santecsoftware.com.ar", # Tu web real
        "http://127.0.0.1:5500",         # Tu VS Code (Live Server)
        "http://localhost:5500"          # Variación local
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de la IA (Busca la clave en el sistema, NO en el código)
api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

# Configuración de cómo responde el modelo
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

# Instrucciones del sistema (Tu "Empleado")
system_instruction = """
Sos el asistente virtual de SanTec Software. 
Tu objetivo es atender a dueños de inmobiliarias y constructoras.
Sos profesional, breve y vas al grano.
Querés vender soluciones de automatización (chatbots, IA, webs).
Si te preguntan precios, decí que depende del proyecto y pedí un contacto.
"""

model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# --- RUTAS DEL SERVIDOR ---

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "Santec Brain Online 🟢"}

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    if not api_key:
        return {"response": "Error: Falta configurar la API KEY en el servidor."}
    
    try:
        chat = model.start_chat(history=[])
        response = chat.send_message(request.message)
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Error del sistema: {str(e)}"}
