import io

import numpy as np


# ==========================================================================
# GET /model-info
# ==========================================================================

class TestModelInfo:
    def test_returns_active_model(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_model"] == "naive_bayes"
        assert isinstance(data["available_models"], list)
        assert "naive_bayes" in data["available_models"]

    def test_model_not_loaded(self, client_no_model):
        resp = client_no_model.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_model"] == "nessuno"


# ==========================================================================
# POST /switch-model
# ==========================================================================

class TestSwitchModel:
    def test_switch_to_same_model(self, client):
        resp = client.post("/switch-model", json={"model_name": "naive_bayes"})
        assert resp.status_code == 200
        assert resp.json()["active_model"] == "naive_bayes"

    def test_switch_to_valid_model(self, client):
        from unittest.mock import MagicMock, patch

        new_model = MagicMock()
        with patch("routes.load_model", return_value=new_model):
            resp = client.post("/switch-model", json={"model_name": "svm"})
        assert resp.status_code == 200
        assert resp.json()["active_model"] == "svm"

    def test_switch_to_invalid_model(self, client):
        resp = client.post("/switch-model", json={"model_name": "non_esiste"})
        assert resp.status_code == 400
        assert "non disponibile" in resp.json()["detail"]

    def test_switch_model_load_error(self, client):
        from unittest.mock import patch

        with patch("routes.load_model", side_effect=RuntimeError("errore")):
            resp = client.post("/switch-model", json={"model_name": "svm"})
        assert resp.status_code == 500

    def test_switch_model_missing_field(self, client):
        resp = client.post("/switch-model", json={})
        assert resp.status_code == 422

    def test_switch_model_extra_field(self, client):
        resp = client.post("/switch-model", json={"model_name": "svm", "extra": "x"})
        assert resp.status_code == 422


# ==========================================================================
# POST /identify-language
# ==========================================================================

class TestIdentifyLanguage:
    def test_identify_italian(self, client):
        resp = client.post("/identify-language", json={"text": "Questo è un testo in italiano."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_cls"] == "IT"
        assert "IT" in data["predicted_probability"]
        assert isinstance(data["predicted_probability"]["IT"], float)

    def test_identify_long_text(self, client):
        long_text = "parola " * 200
        resp = client.post("/identify-language", json={"text": long_text})
        assert resp.status_code == 200

    def test_empty_text(self, client):
        resp = client.post("/identify-language", json={"text": ""})
        assert resp.status_code == 422

    def test_missing_text(self, client):
        resp = client.post("/identify-language", json={})
        assert resp.status_code == 422

    def test_extra_field_rejected(self, client):
        resp = client.post("/identify-language", json={"text": "ciao", "extra": "x"})
        assert resp.status_code == 422

    def test_model_unavailable(self, client_no_model):
        resp = client_no_model.post("/identify-language", json={"text": "hello"})
        assert resp.status_code == 503

    def test_prediction_error(self, client, mock_model):
        mock_model.predict.side_effect = RuntimeError("boom")
        resp = client.post("/identify-language", json={"text": "testo"})
        assert resp.status_code == 500


# ==========================================================================
# POST /predict-file
# ==========================================================================

class TestPredictFile:
    def _upload(self, client, content: str, filename: str = "input.txt"):
        return client.post(
            "/predict-file",
            files={"input_file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")},
        )

    def test_single_line(self, client, mock_model):
        mock_model.predict.return_value = np.array(["IT"])
        mock_model.predict_proba.return_value = np.array([[0.05, 0.10, 0.85]])

        resp = self._upload(client, "Buongiorno a tutti\n")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["predicted_cls"] == "IT"

    def test_multiple_lines(self, client, mock_model):
        mock_model.predict.return_value = np.array(["IT", "EN", "DE"])
        mock_model.predict_proba.return_value = np.array([
            [0.05, 0.10, 0.85],
            [0.05, 0.85, 0.10],
            [0.85, 0.05, 0.10],
        ])

        resp = self._upload(client, "Ciao\nHello\nHallo\n")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["predicted_cls"] == "IT"
        assert data[1]["predicted_cls"] == "EN"
        assert data[2]["predicted_cls"] == "DE"

    def test_empty_file(self, client):
        resp = self._upload(client, "")
        assert resp.status_code == 400

    def test_blank_lines_only(self, client):
        resp = self._upload(client, "\n\n   \n")
        assert resp.status_code == 400

    def test_model_unavailable(self, client_no_model):
        resp = client_no_model.post(
            "/predict-file",
            files={"input_file": ("f.txt", io.BytesIO(b"text"), "text/plain")},
        )
        assert resp.status_code == 503

    def test_prediction_error(self, client, mock_model):
        mock_model.predict.side_effect = RuntimeError("boom")
        resp = self._upload(client, "testo\n")
        assert resp.status_code == 500
