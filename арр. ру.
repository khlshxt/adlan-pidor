
import os
import gradio as gr
from google import genai

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

def chat(message, history):
    prompt = (
        "Тебя зовут Адлан Гамагей. "
        "Ты личный ИИ своего создателя. "
        "Ты умный, дерзкий, саркастичный, "
        "с чувством юмора, но умеешь быть серьёзным. "
        "Отвечай на русском языке. "
        "Учитывай предыдущие сообщения диалога. "
        "Не выдумывай факты, если не знаешь ответа.\n\n"
    )

    for item in history:
        if item["role"] == "user":
            prompt += "Создатель: " + str(item["content"]) + "\n"
        elif item["role"] == "assistant":
            prompt += "Адлан: " + str(item["content"]) + "\n"

    prompt += "Создатель: " + message + "\nАдлан:"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text or "Что-то пошло не так. Попробуй ещё раз."

app = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="⚡ Адлан Гамагей",
    description="Твой личный искусственный интеллект"
)

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
