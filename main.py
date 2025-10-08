import streamlit as st
import sys
import random
from transformers import pipeline
from langdetect import detect, DetectorFactory

sys.path.append('/home/fedosdan2/prog/pr_act/GenAI-3-03/GenAi-1-05-chat_bot')
sys.path.append('/home/fedosdan2/prog/pr_act/GenAI-3-03/third_course_ML/code/Block1/GenAI-1-06')
import pac1
import task 

# Глобальные константы
PREDEFINED_RESPONSES_RU = [
    "Мне очень жаль, что вы так себя чувствуете. Давайте поговорим о чём-нибудь хорошем! Например, что вас радовало на этой неделе?",
    "Спасибо, что поделились. Иногда помогает сменить фокус — а вы любите смотреть закаты или рассветы?",
    "Я вас слышу. А знаете, что сегодня в мире произошло что-то доброе? Хотите, расскажу?",
    "Всё временно — и плохое, и хорошее. А вы верите, что завтра будет лучше?",
    "Иногда всё кажется серым, но я верю, что впереди светлое! Хотите обсудить что-то вдохновляющее?"
]

PREDEFINED_RESPONSES_EN = [
    "I'm really sorry you're feeling this way. Let's talk about something good! For example, what made you happy this week?",
    "Thank you for sharing. Sometimes it helps to shift focus — do you enjoy watching sunsets or sunrises?",
    "I hear you. Did you know something kind happened in the world today? Would you like me to tell you about it?",
    "Everything is temporary — both the bad and the good. Do you believe tomorrow will be better?",
    "Sometimes everything seems gray, but I believe brighter days are ahead! Would you like to discuss something inspiring?"
]

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

CLASSIFIER_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

MAX_LEN = 512


@st.cache_resource
def load_translation_pipelines():
    """Загружает и кэширует модели машинного перевода для работы с русским и английским языками.

    Returns:
        tuple: Кортеж из двух объектов pipeline:
            - en2ru (pipeline): Модель перевода с английского на русский.
            - ru2en (pipeline): Модель перевода с русского на английский.
            В случае ошибки возвращает (None, None).
    """

    try:
        en2ru = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru", device=-1)
        ru2en = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en", device=-1)
        return en2ru, ru2en
    except Exception as e:
        st.error(f"Ошибка загрузки моделей перевода: {e}")
        return None, None


def safe_detect_language(text: str) -> str:
    """Безопасно определяет язык текста с обработкой исключений.

    Args:
        text (str): Входной текст для анализа.

    Returns:
        str: Код языка ('en' или 'ru'). При ошибке или неизвестном языке возвращает 'en'.
    """

    try:
        lang = detect(text)
        return lang if lang in ["en", "ru"] else "en"
    except:
        return "en"


def safe_translate(text: str, translator, src_lang: str, tgt_lang: str) -> str:
    """Безопасно выполняет перевод текста с обработкой ошибок и ограничением длины.

    Args:
        text (str): Текст для перевода.
        translator (pipeline): Объект модели перевода из transformers.
        src_lang (str): Код исходного языка (например, 'en').
        tgt_lang (str): Код целевого языка (например, 'ru').

    Returns:
        str: Переведённый текст. В случае ошибки возвращает исходный текст.
    """

    try:
        # Ограничиваем длину текста (модель не любит длинные строки)
        max_len = MAX_LEN
        if len(text) > max_len:
            text = text[:max_len]
        result = translator(text)
        return result[0]['translation_text']
    except Exception as e:
        st.warning(f"Ошибка перевода ({src_lang}→{tgt_lang}): {str(e)[:100]}")
        return text  # возвращаем оригинал


@st.cache_resource
def load_chatbot():
    """Загружает и кэширует модель чат-бота и её токенизатор.

    Returns:
        tuple: Кортеж (chatbot, tokenizer) или (None, None) в случае ошибки.
    """

    chatbot, tokenizer = pac1.create_chatbot(model_name=MODEL_NAME, device=-1)
    if chatbot is None:
        st.error("Не удалось загрузить модель чата.")
        return None, None
    return chatbot, tokenizer


@st.cache_resource
def load_classifier():
    """Загружает и кэширует модель анализа тональности.

    Returns:
        pipeline or None: Объект классификатора или None в случае ошибки.
    """

    classifier = task.load_classifier(CLASSIFIER_NAME)
    if classifier is None:
        st.error("Не удалось загрузить классификатор тональности.")
        return None
    return classifier


def generate_topic_shift_reply(chatbot, tokenizer, history, user_input):
    """Генерирует ответ, который мягко меняет тему при негативной тональности.

    Args:
        chatbot (pipeline): Модель генерации текста.
        tokenizer (AutoTokenizer): Токенизатор модели.
        history (list[dict]): История диалога в формате [{"role": "...", "content": "..."}, ...].
        user_input (str): Последнее сообщение пользователя.

    Returns:
        str: Сгенерированный ответ бота без промпта и стоп-токенов.
    """

    messages = history + [
        {"role": "user", "content": user_input},
        {"role": "system", "content": "The user seems upset. Respond kindly and gently change the topic to something positive or neutral."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = chatbot(
        prompt,
        max_new_tokens=MAX_LEN,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = result[0]["generated_text"]
    reply = generated[len(prompt):]
    for stop in ["<|user|>", "<|system|>", "<|assistant|>", "</s>"]:
        if stop in reply:
            reply = reply.split(stop)[0]
            break
    return reply.strip()


def main():
    """Основная функция запуска Streamlit-приложения чат-бота с поддержкой двух языков.

    Приложение:
    - Определяет язык ввода пользователя (русский/английский)
    - Анализирует тональность сообщения
    - При негативной тональности предлагает сменить тему
    - Поддерживает два режима ответа: динамическая генерация и заготовленные фразы
    - Автоматически переводит ответы на язык пользователя
    """

    st.set_page_config(page_title="Ask GigaChat", page_icon="💬", layout="centered")
    st.title("💬 Ask GigaChat")
    st.markdown("Дружелюбный ассистент с поддержкой русского и английского языков")

    use_dynamic = st.toggle("Динамическая генерация", value=True)

    # Загружаем ВСЕ модели
    chatbot, tokenizer = load_chatbot()
    classifier = load_classifier()
    en2ru, ru2en = load_translation_pipelines()
    
    if None in [chatbot, classifier, en2ru, ru2en]:
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Напишите сообщение...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        # Определяем язык пользователя
        user_lang = safe_detect_language(prompt)
        st.caption(f"Язык: {user_lang}")

        # Анализ тональности (модель работает на английском, поэтому переводим при необходимости)
        analysis_text = prompt
        if user_lang == "ru":
            analysis_text = safe_translate(prompt, ru2en, "ru", "en")
        
        sentiment, score = task.analyze_sentiment(analysis_text, classifier)
        st.caption(f"Тональность: {sentiment} (уверенность: {score:.2f})")

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Думаю... 🤔")

            if sentiment == "negative":
                if use_dynamic:
                    # Генерируем ответ на АНГЛИЙСКОМ (модель TinyLlama — английская)
                    bot_reply_en = generate_topic_shift_reply(chatbot, tokenizer, st.session_state.history, analysis_text)
                    # Переводим на язык пользователя
                    bot_reply = safe_translate(bot_reply_en, en2ru, "en", "ru") if user_lang == "ru" else bot_reply_en
                else:
                    # Используем заготовленную фразу на нужном языке
                    bot_reply = random.choice(PREDEFINED_RESPONSES_RU if user_lang == "ru" else PREDEFINED_RESPONSES_EN)
            else:
                # Обычный ответ
                bot_reply_en = pac1.get_bot_reply(chatbot, tokenizer, st.session_state.history, analysis_text)
                bot_reply = safe_translate(bot_reply_en, en2ru, "en", "ru") if user_lang == "ru" else bot_reply_en

            message_placeholder.markdown(bot_reply)

        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": bot_reply})


if __name__ == "__main__":
    main()