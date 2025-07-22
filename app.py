"""
Aplicación web con Flask. Agente de IA experto basado en un documento PDF.
El agente responderá únicamente con la información contenida en el PDF proporcionado.
"""

import os
import markdown
import fitz  # PyMuPDF
from flask import Flask, request, render_template, session
from flask_session import Session
import google.generativeai as genai

# --- NUEVO: Función para extraer texto de un PDF ---
def extract_pdf_text(pdf_path):
    """Lee un archivo PDF y devuelve todo su texto como una sola cadena."""
    if not os.path.exists(pdf_path):
        return "Error: No se encontró el archivo PDF en la ruta especificada."
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return f"Error al leer el archivo PDF: {e}"

# --- 1. Configuración de la aplicación Flask ---
app = Flask(__name__)
app.secret_key = "una-clave-secreta-muy-segura"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- 2. Configuración del modelo Gemini y carga de datos ---
COMPANY_DATA_PDF = "empresa_data.pdf"
COMPANY_KNOWLEDGE = extract_pdf_text(COMPANY_DATA_PDF) # El conocimiento se carga una sola vez al iniciar

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró la variable de entorno GEMINI_API_KEY.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Error al configurar Gemini: {e}")
    model = None

MAX_HISTORY = 3

@app.route("/")
def home():
    if "history" not in session:
        session["history"] = []
    return render_template("index.html", history=session["history"])


@app.route("/predict", methods=["POST"])
def predict():
    if not model or "Error" in COMPANY_KNOWLEDGE:
        error_msg = "El modelo de IA no está configurado o no se pudo leer el PDF. Revisa la consola."
        return render_template("index.html", error=error_msg, history=session.get("history", []))

    prompt = request.form.get("prompt")
    if not prompt:
        return render_template("index.html", error="Por favor, ingresa un texto válido.", history=session.get("history", []))

    history = session.get("history", [])
    chat_history_context = ""
    for item in history[-MAX_HISTORY:]:
        chat_history_context += f"Usuario: {item['prompt']}\nModelo: {item['response_raw']}\n"

    # --- NUEVO: Construcción del Prompt Especializado ---
    # Este es el cambio más importante. Creamos un "mega-prompt" que le dice a la IA exactamente cómo comportarse.
    specialized_prompt = f"""
    Eres un asistente de IA experto para la empresa "Datacen". Tu única fuente de conocimiento es el siguiente texto.
    No debes responder preguntas que no se puedan contestar basándote en este texto.
    Si la respuesta no está en el texto, responde cortésmente: "No tengo información sobre ese tema".
    No inventes información. Sé conciso y directo.

    --- INICIO DEL DOCUMENTO DE CONOCIMIENTO ---
    {COMPANY_KNOWLEDGE}
    --- FIN DEL DOCUMENTO DE CONOCIMIENTO ---

    Considerando la conversación previa:
    {chat_history_context}

    Responde a la siguiente pregunta del usuario basándote ESTRICTAMENTE en el documento de conocimiento:
    Usuario: {prompt}
    """

    try:
        response = model.generate_content(specialized_prompt)
        response_text = response.text
        response_html = markdown.markdown(response_text)

        history.append({
            "prompt": prompt,
            "response_raw": response_text,
            "response_html": response_html
        })
        session["history"] = history

        return render_template("index.html", prompt=prompt, response_html=response_html, history=history)

    except Exception as e:
        error_message = f"Ocurrió un error al contactar al modelo de IA: {e}"
        return render_template("index.html", error=error_message, history=history)


if __name__ == "__main__":
    app.run(debug=True)
    