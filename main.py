import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
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

# --- CONFIGURACIÓN DEL MODELO ---
# Intentamos usar el más nuevo, pero si falla, no rompemos nada todavía.
MODELO_ELEGIDO = "gemini-1.5-flash" 

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    if not api_key:
        return {"response": "Error: Falta la API KEY en Render."}

    try:
        # Intentamos crear el modelo
        model = genai.GenerativeModel(MODELO_ELEGIDO)
        chat = model.start_chat(history=[])
        response = chat.send_message(request.message)
        return {"response": response.text}

    except Exception as e:
        # SI FALLA, HACEMOS ESTO:
        error_msg = str(e)
        
        # Si es el error 404 maldito, buscamos qué modelos SÍ hay
        if "404" in error_msg or "not found" in error_msg:
            try:
                lista_modelos = []
                for m in genai.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        lista_modelos.append(m.name)
                
                return {"response": f"⚠️ EL MODELO '{MODELO_ELEGIDO}' NO EXISTE PARA TU CUENTA. \n\n✅ USA UNO DE ESTOS (Copiame el nombre y lo cambiamos):\n" + "\n".join(lista_modelos)}
            except Exception as e2:
                return {"response": f"Error doble: {str(e2)}"}
        
        return {"response": f"Error del sistema: {error_msg}"}
