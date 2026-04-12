import sys
import os

# Aggiungi src al path PRIMA di qualsiasi import dal progetto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix
from MuseumLangAPI import app


# ---------------------------------------------------------------------------
# Mock del vectorizer TF-IDF
# ---------------------------------------------------------------------------

def _make_mock_vectorizer():
    """Crea un vectorizer mock che restituisce matrici sparse."""
    vectorizer = MagicMock()
    # transform restituisce una matrice sparsa (come il TfidfVectorizer reale)
    vectorizer.transform.side_effect = lambda texts: csr_matrix(np.ones((len(texts), 10)))
    return vectorizer


# ---------------------------------------------------------------------------
# Mock del modello
# ---------------------------------------------------------------------------

def _make_mock_model():
    """Crea un modello mock che simula predict, predict_proba e classes_."""
    model = MagicMock()
    model.classes_ = np.array(["DE", "EN", "IT"])
    model.predict.side_effect = lambda x: np.array(["IT"] * x.shape[0])
    model.predict_proba.side_effect = lambda x: np.array([[0.05, 0.10, 0.85]] * x.shape[0])
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_model():
    return _make_mock_model()


@pytest.fixture()
def mock_vectorizer():
    return _make_mock_vectorizer()


@pytest.fixture()
def client(mock_model, mock_vectorizer):
    """TestClient con modello e vectorizer mock iniettati, senza caricare pickle reali."""
    with patch("MuseumLangAPI.load_model", return_value=mock_model), \
            patch("MuseumLangAPI.load_vectorizer", return_value=mock_vectorizer), \
            patch("routes.load_model", return_value=mock_model):
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def client_no_model():
    """TestClient con modello e vectorizer non disponibili (None)."""
    with patch("MuseumLangAPI.load_model", side_effect=Exception("model not found")), \
            patch("MuseumLangAPI.load_vectorizer", side_effect=Exception("vectorizer not found")):
        with TestClient(app) as c:
            yield c
