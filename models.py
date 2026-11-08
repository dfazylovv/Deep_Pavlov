from deeppavlov import build_model


# Модели загружаются только при первом обращении к соответствующей функции.
ner_model = None
sentiment_model = None
qa_model = None


def get_ner_model():
    global ner_model
    if ner_model is None:
        print("Загрузка модели NER...")
        ner_model = build_model(
            "ner_rus_bert",
            download=True,
            install=False
        )
    return ner_model


def get_sentiment_model():
    global sentiment_model
    if sentiment_model is None:
        print("Загрузка модели анализа тональности...")
        sentiment_model = build_model(
            "rusentiment_bert",
            download=True,
            install=False
        )
    return sentiment_model


def get_qa_model():
    global qa_model
    if qa_model is None:
        print("Загрузка модели Question Answering...")
        qa_model = build_model(
            "squad_ru_bert",
            download=True,
            install=False
        )
    return qa_model


def recognize_entities(text):
    """Распознавание людей, организаций и местоположений."""
    model = get_ner_model()
    result = model([text])

    tokens = result[0][0]
    tags = result[1][0]

    entity_names = {
        "PER": "Человек",
        "ORG": "Организация",
        "LOC": "Местоположение"
    }

    entities = []
    current_tokens = []
    current_type = None

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_tokens:
                entities.append(
                    (entity_names.get(current_type, current_type),
                     " ".join(current_tokens))
                )

            current_type = tag[2:]
            current_tokens = [token]

        elif tag.startswith("I-") and current_tokens:
            current_tokens.append(token)

        else:
            if current_tokens:
                entities.append(
                    (entity_names.get(current_type, current_type),
                     " ".join(current_tokens))
                )
                current_tokens = []
                current_type = None

    if current_tokens:
        entities.append(
            (entity_names.get(current_type, current_type),
             " ".join(current_tokens))
        )

    if not entities:
        return "Именованные сущности не обнаружены."

    return "\n".join(
        f"• {name} — {entity_type}"
        for entity_type, name in entities
    )


def analyze_sentiment(text):
    """Определение эмоциональной окраски текста."""
    model = get_sentiment_model()
    result = model([text])

    sentiment = str(result[0]).lower()

    descriptions = {
        "positive": "Положительная 😊",
        "negative": "Отрицательная 😟",
        "neutral": "Нейтральная 😐"
    }

    return descriptions.get(sentiment, f"Результат модели: {sentiment}")


def answer_question(context, question):
    """Поиск ответа на вопрос в переданном тексте."""
    model = get_qa_model()

    result = model([context], [question])

    answer = result[0][0]
    confidence = float(result[2][0])

    return (
        f"Ответ: {answer}\n"
        f"Уверенность модели: {confidence:.4f}"
    )
