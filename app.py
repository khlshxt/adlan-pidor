import os
import gradio as gr
from google import genai
from google.genai import types

# Инициализация клиента
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

def chat(message, history):
    # Транслируем историю под формат Gemini (user -> user, assistant -> model)
    formatted_contents = []
    for item in history:
        role = "user" if item["role"] == "user" else "model"
        formatted_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=item["content"])]
            )
        )
    
    # Добавляем реплику пользователя
    formatted_contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)]
        )
    )
    
    # Задаем системные инструкции
    system_instruction = (
        "Тебя зовут Адлан Гамагей. "
        "Ты дерзкий, умный и саркастичный ИИ. "
        "Обращайся к пользователю 'Создатель'. "
        "Отвечай строго на русском языке."
    )
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.8
    )
    
    # Используем актуальную и быструю модель gemini-2.5-flash
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=formatted_contents,
        config=config
    )
    
    return response.text

app = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="⚡ Адлан Гамагей",
    description="Твой личный искусственный интеллект"
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )


