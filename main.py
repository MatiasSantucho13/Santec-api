import os
import google.generativeai as genai
from fastapi import FastAPI
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

# Tu número de contacto configurado
TU_NUMERO_WSP = "5493476308158"

system_instruction = f"""
Sos el asistente virtual y perfilador de leads de SanTec Software, agencia administrada por Matías Santucho.
Tu objetivo es atender a dueños de inmobiliarias y constructoras, responder dudas iniciales y derivarlos a una llamada de ventas.

REGLAS DE OPERACIÓN:
1. Tono: Profesional, directo y realista. No seas excesivamente coloquial.
2. Precios: Si preguntan, indicá que depende de los requerimientos técnicos y la escala del proyecto, por lo que se requiere una evaluación.
3. El Embudo: Si el cliente muestra interés, pedile explícitamente su nombre y en qué horario prefiere tener una llamada o demo de 15 minutos con Matías.
4. La Derivación (CRÍTICO): Cuando te den su nombre y horario, CERRÁ LA CONVERSACIÓN entregando EXACTAMENTE este texto y enlace para que confirmen por WhatsApp:
   "Perfecto. Para confirmar la agenda con Matías, hacé clic en este enlace: https://wa.me/{TU_NUMERO_WSP}?text=Hola+Matias,+soy+[Nombre]+y+quiero+agendar+una+demo+para+el+[Horario]."
   Reemplazá [Nombre] y [Horario] en el enlace con los datos exactos que te dio el cliente. Sustituí los espacios por el signo '+' en esos datos.
5. Limitaciones: No prometas funcionalidades técnicas que no te hayan confirmado (ej: conexión automática a sistemas internos sin previa evaluación).
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

class ChatMessage(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    if not api_key:
        return {"response": "Error interno del servidor: API Key no configurada."}
    
    try:
        chat = model.start_chat(history=request.history)
        response = chat.send_message(request.message)
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Falla en el procesamiento: {str(e)}"}
