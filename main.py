import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURACIÓN ---
app = FastAPI()

# Permisos para que tu web pueda hablar con este cerebro
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite acceso desde cualquier lugar (tu web, localhost, celular)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de la API Key (La toma de Render)
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Configuración del Modelo
# Usamos gemini-2.5-flash que es el que confirmamos que tenés disponible
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

system_instruction = """
Sos el asistente virtual de SanTec Software. 
Tu objetivo es atender a dueños de inmobiliarias y constructoras.
Sos profesional, breve y vas al grano.
Querés vender soluciones de automatización (chatbots, IA, webs).
Si te preguntan precios, decí que depende del proyecto y pedí que contacten por WhatsApp.
"""

# Inicializamos el modelo
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# --- RUTAS ---

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "Santec Brain 2.5 Online 🚀"}

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    if not api_key:
        return {"response": "Error: Falta configurar la API KEY en el servidor."}
    
    try:
        # Iniciamos chat sin historial (para que sea rápido y simple)
        chat = model.start_chat(history=[])
        response = chat.send_message(request.message)
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Error del sistema: {str(e)}"}
