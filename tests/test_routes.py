import io


# ===========================================================================
# GET /model-info
# ===========================================================================

class TestModelInfo:

    def test_model_info_ok(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_model"] == "naive_bayes"
        assert "naive_bayes" in data["available_models"]

    def test_model_info_no_model(self, client_no_model):
        resp = client_no_model.get("/model-info")
        assert resp.status_code == 200
        assert resp.json()["active_model"] == "nessuno"


# ===========================================================================
# POST /switch-model
# ===========================================================================

class TestSwitchModel:

    def test_switch_model_ok(self, client):
        resp = client.post("/switch-model", json={"model_name": "svm"})
        assert resp.status_code == 200
        assert resp.json()["active_model"] == "svm"

    def test_switch_model_gia_attivo(self, client):
        resp = client.post("/switch-model", json={"model_name": "naive_bayes"})
        assert resp.status_code == 200
        assert resp.json()["active_model"] == "naive_bayes"

    def test_switch_model_non_esistente(self, client):
        resp = client.post("/switch-model", json={"model_name": "modello_fake"})
        assert resp.status_code == 400
        assert "non disponibile" in resp.json()["detail"]

    def test_switch_model_payload_vuoto(self, client):
        resp = client.post("/switch-model", json={})
        assert resp.status_code == 422


# ===========================================================================
# POST /identify-language
# ===========================================================================

class TestIdentifyLanguage:

    def test_identify_language_ok(self, client):
        resp = client.post("/identify-language", json={"text": "Questo è un testo in italiano."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_cls"] == "IT"
        assert "IT" in data["predicted_probability"]
        assert "EN" in data["predicted_probability"]
        assert "DE" in data["predicted_probability"]

    def test_identify_language_risposta_probabilita(self, client):
        resp = client.post("/identify-language", json={"text": "Ciao mondo"})
        data = resp.json()
        proba = data["predicted_probability"]
        assert abs(sum(proba.values()) - 1.0) < 0.01

    def test_identify_language_testo_vuoto(self, client):
        resp = client.post("/identify-language", json={"text": ""})
        assert resp.status_code == 422

    def test_identify_language_campo_mancante(self, client):
        resp = client.post("/identify-language", json={})
        assert resp.status_code == 422

    def test_identify_language_campo_extra(self, client):
        resp = client.post("/identify-language", json={"text": "ciao", "extra": "valore"})
        assert resp.status_code == 422

    def test_identify_language_modello_non_disponibile(self, client_no_model):
        resp = client_no_model.post("/identify-language", json={"text": "test"})
        assert resp.status_code == 503

    def test_identify_language_testo_con_punteggiatura(self, client):
        resp = client.post("/identify-language", json={"text": "Hello, world! 123."})
        assert resp.status_code == 200
        assert resp.json()["predicted_cls"] == "IT"  # il mock ritorna sempre IT

    def test_identify_language_testo_lungo(self, client):
        long_text = "parola " * 500
        resp = client.post("/identify-language", json={"text": long_text})
        assert resp.status_code == 200


# ===========================================================================
# POST /predict-file
# ===========================================================================

class TestPredictFile:

    def _upload(self, client, content: str, filename: str = "test.txt"):
        return client.post(
            "/predict-file",
            files={"input_file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")},
        )

    def test_predict_file_ok(self, client):
        resp = self._upload(client, "Riga in italiano\nA row in English\nEine Zeile auf Deutsch")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        for item in data:
            assert "predicted_cls" in item
            assert "predicted_probability" in item

    def test_predict_file_singola_riga(self, client):
        resp = self._upload(client, "Solo una riga")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_predict_file_vuoto(self, client):
        resp = self._upload(client, "")
        assert resp.status_code == 400
        assert "vuoto" in resp.json()["detail"]

    def test_predict_file_solo_righe_vuote(self, client):
        resp = self._upload(client, "\n\n\n")
        assert resp.status_code == 400

    def test_predict_file_modello_non_disponibile(self, client_no_model):
        resp = self._upload(client_no_model, "qualche testo")
        assert resp.status_code == 503

    def test_predict_file_ignora_righe_vuote(self, client):
        resp = self._upload(client, "riga uno\n\nriga due\n\n")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_predict_file_probabilita(self, client):
        resp = self._upload(client, "test line")
        data = resp.json()
        proba = data[0]["predicted_probability"]
        assert abs(sum(proba.values()) - 1.0) < 0.01
