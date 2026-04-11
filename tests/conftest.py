import sys
import os

# Aggiungi src al path PRIMA di qualsiasi import dal progetto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from MuseumLangAPI import app


def _make_mock_model():
    """Crea un modello mock che simula predict, predict_proba e classes_."""
    model = MagicMock()
    model.classes_ = np.array(["DE", "EN", "IT"])
    model.predict.return_value = np.array(["IT"])
    model.predict_proba.return_value = np.array([[0.05, 0.10, 0.85]])
    return model


@pytest.fixture()
def mock_model():
    return _make_mock_model()


@pytest.fixture()
def client(mock_model):
    """TestClient con modello mock iniettato, senza caricare pickle reali."""
    with patch("MuseumLangAPI.load_model", return_value=mock_model):
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def client_no_model():
    """TestClient con modello non disponibile (None)."""
    with patch("MuseumLangAPI.load_model", side_effect=Exception("model not found")):
        with TestClient(app) as c:
            yield c
