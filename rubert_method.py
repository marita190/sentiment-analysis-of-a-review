from storage import finalize_result

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
_MODEL = None
_TOKENIZER = None


def _load_model():
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise ImportError(
            "Для RuBERT нужны пакеты transformers и torch. "
            "Установи: pip install transformers torch"
        ) from e

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _MODEL.eval()

    return _TOKENIZER, _MODEL


def analyze_with_rubert(text):
    tokenizer, model = _load_model()

    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    id2label = model.config.id2label
    if not id2label:
        id2label = {
            0: "NEUTRAL",
            1: "POSITIVE",
            2: "NEGATIVE"
        }

    scores = {}
    for index, probability in enumerate(probabilities):
        label = id2label.get(index, str(index)).lower()
        scores[label] = float(probability)

    sentiment = max(scores, key=scores.get)

    if "pos" in sentiment:
        final_sentiment = "positive"
        emoji = "😊"
        color = "green"
    elif "neg" in sentiment:
        final_sentiment = "negative"
        emoji = "😞"
        color = "red"
    else:
        final_sentiment = "neutral"
        emoji = "😐"
        color = "gray"

    keywords = [
        f"{k}:{round(v, 3)}"
        for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]

    result = {
        "sentiment": final_sentiment,
        "emoji": emoji,
        "color": color,
        "score": round(scores[sentiment], 2),
        "keywords": keywords[:5]
    }

    return finalize_result(result, text)
