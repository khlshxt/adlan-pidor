import os
import gradio as gr
from google import genai
from google.genai import types

# Инициализация клиента
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

def chat(message, history):
    # 1. Формируем историю в родном формате Gemini API
    formatted_contents = []
    
    for item in history:
        # Gradio type="messages" передает роли "user" и "assistant"
        role = "user" if item["role"] == "user" else "model"
        formatted_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=item["content"])]
            )
        )
    
    # 2. Добавляем последнее сообщение пользователя
    formatted_contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)]
        )
    )
    
    # 3. Задаем системные инструкции для кастомизации личности
    system_instruction = (
        "Тебя зовут Адлан Гамагей. "
        "Ты дерзкий, умный и саркастичный ИИ. "
        "Обращайся к пользователю 'Создатель'. "
        "Отвечай строго на русском языке."
    )
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7 # Немного творчества для сарказма
    )
    
    # 4. Делаем запрос к модели
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=formatted_contents,
        config=config
    )
    
    return response.text

# Инициализация интерфейса Gradio
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

