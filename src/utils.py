import os
import pickle
import re
import string
from typing import List

from config import AVAILABLE_MODELS, VECTORIZER_PATH


def load_model(model_name: str):
    """Carica un modello pickle dal registro AVAILABLE_MODELS."""
    model_path = AVAILABLE_MODELS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File modello non trovato: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_vectorizer():
    """Carica il vectorizer TF-IDF salvato durante il training."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer non trovato: {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def data_cleaner(texts: List[str]) -> List[str]:
    """
    Preprocessing dei testi: rimozione di punteggiatura, numeri e spazi extra.
    I testi vengono convertiti in minuscolo per uniformità.
    """
    cleaned = []
    for sentence in texts:
        sentence = sentence.lower()
        for c in string.punctuation:
            sentence = sentence.replace(c, " ")
        sentence = re.sub(r'\d+', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        cleaned.append(sentence)
    return cleaned


def preprocess_text(text: str) -> str:
    """Applica il preprocessing a un singolo testo."""
    return data_cleaner([text])[0]
