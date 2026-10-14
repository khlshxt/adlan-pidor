import os
import gradio as gr
from google import genai
from google.genai import types

# Инициализация клиента
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def _part_from_text(text: str):
    """Универсальная обёртка для создания Part — использует types.Part, если доступен."""
    try:
        return types.Part.from_text(text=text)
    except Exception:
        # Упрощённый fallback в случае несовместимости версии SDK
        return {"type": "input_text", "text": text}


def _content(role: str, text: str):
    """Создаёт объект Content совместимо с разными версиями SDK."""
    try:
        return types.Content(role=role, parts=[_part_from_text(text)])
    except Exception:
        # Фallback: словарь в формате, который иногда принимают HTTP клиенты
        return {"role": role, "content": [{"type": "input_text", "text": text}]}


def _extract_text_from_response(response) -> str | None:
    """Пытается извлечь текст из ответа модели в нескольких вариантах."""
    # В некоторых версиях SDK есть атрибут text
    if hasattr(response, "text") and response.text:
        return response.text

    # Попытка через candidates -> content / output -> content -> parts / text
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            cand = candidates[0]
            # cand.output -> list -> item.content -> list -> part.text
            out = getattr(cand, "output", None)
            if out:
                maybe = out[0]
                content = getattr(maybe, "content", None)
                if content:
                    part = content[0]
                    if hasattr(part, "text") and part.text:
                        return part.text
                    # иногда parts внутри content
                    parts = getattr(part, "parts", None)
                    if parts and len(parts) > 0:
                        # parts могут быть Part-объектами или dict
                        p0 = parts[0]
                        return getattr(p0, "text", None) or p0.get("text")
            # cand.content -> list -> part.text
            content2 = getattr(cand, "content", None)
            if content2:
                part = content2[0]
                return getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
    except Exception:
        pass

    # Иногда SDK возвращает dict-представление
    try:
        rd = response.to_dict() if hasattr(response, "to_dict") else dict(response)
        # deep lookup
        if "candidates" in rd and rd["candidates"]:
            cand = rd["candidates"][0]
            # try several nested keys
            for key in ("output", "content", "message", "text"):
                if key in cand:
                    # if 'text' directly
                    if key == "text":
                        return cand["text"]
                    val = cand[key]
                    if isinstance(val, list) and val:
                        # drill into first element
                        first = val[0]
                        if isinstance(first, dict):
                            for k in ("text",):
                                if k in first and first[k]:
                                    return first[k]
    except Exception:
        pass

    return None


def chat(message, history):
    # 1. Формируем историю в родном формате Gemini API (или fallback словарях)
    formatted_contents = []

    # Gradio type="messages" передаёт роли "user" и "assistant"
    for item in history or []:
        role = "user" if item.get("role") == "user" else "model"
        formatted_contents.append(_content(role, str(item.get("content", ""))))

    # 2. Добавляем последнее сообщение пользователя
    formatted_contents.append(_content("user", str(message)))

    # 3. Системные инструкции для кастомизации личности
    system_instruction = (
        "Тебя зовут Адлан Гамагей. "
        "Ты дерзкий, умный и саркастичный ИИ. "
        "Обращайся к пользователю 'Создатель'. "
        "Отвечай строго на русском языке."
    )

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,  # Немного творчества для сарказма
        )
    except Exception:
        # Если в SDK нет GenerateContentConfig — используем None и полагаемся на defaults
        config = None

    # 4. Делаем запрос к модели
    try:
        if config is not None:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=formatted_contents,
                config=config,
            )
        else:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=formatted_contents,
            )
    except Exception as e:
        return f"Ошибка при обращении к модели: {e}"

    # 5. Извлекаем текст из ответа в устойчивом виде
    text = _extract_text_from_response(response)
    if not text:
        return "Что-то пошло не так. Попробуй ещё раз."

    return text


# Инициализация интерфейса Gradio
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
