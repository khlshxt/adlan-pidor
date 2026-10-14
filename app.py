import os
import gradio as gr
from google import genai

# Инициализация клиента
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


def build_prompt(message: str, history):
    """
    Собирает историю чата в простой текстовый промпт.
    """
    prompt = (
        "Тебя зовут Адлан Гамагей. "
        "Ты личный ИИ своего создателя. "
        "Ты умный, дерзкий, саркастичный, но можешь быть серьёзным. "
        "Отвечай на русском языке. "
        "Учитывай предыдущие сообщения диалога. "
        "Не выдумывай факты, если не знаешь ответа.\n\n"
    )

    for item in history or []:
        role = item.get("role", "user")
        content = str(item.get("content", ""))
        if role == "user":
            prompt += f"Создатель: {content}\n"
        elif role == "assistant":
            prompt += f"Адлан: {content}\n"

    prompt += f"Создатель: {message}\nАдлан:"
    return prompt


def chat(message, history):
    if not message or not message.strip():
        return "Напиши что-нибудь, Создатель."

    prompt = build_prompt(message, history)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # более стабильная модель для текущего SDK
            contents=prompt,
        )
    except Exception as e:
        return f"Ошибка при обращении к модели: {e}"

    try:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = getattr(candidate.content, "parts", None)
                if parts:
                    texts = []
                    for p in parts:
                        if hasattr(p, "text") and p.text:
                            texts.append(p.text)
                    if texts:
                        return "".join(texts)
        return "Что-то пошло не так. Попробуй ещё раз."
    except Exception:
        return "Что-то пошло не так. Попробуй ещё раз."


app = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="⚡ Адлан Гамагей",
    description="Твой личный искусственный интеллект",
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
)
