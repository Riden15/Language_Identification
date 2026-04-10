import os
import pickle

from config import AVAILABLE_MODELS


def load_model(model_name: str):
    """Carica un modello pickle dal registro AVAILABLE_MODELS."""
    model_path = AVAILABLE_MODELS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File modello non trovato: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)
