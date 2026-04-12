from utils import data_cleaner, preprocess_text


class TestDataCleaner:
    """Test per la funzione data_cleaner."""

    def test_lowercase(self):
        result = data_cleaner(["HELLO WORLD"])
        assert result == ["hello world"]

    def test_rimuove_punteggiatura(self):
        result = data_cleaner(["Ciao, mondo! Come stai?"])
        assert "," not in result[0]
        assert "!" not in result[0]
        assert "?" not in result[0]

    def test_rimuove_numeri(self):
        result = data_cleaner(["Il dipinto risale al 1500"])
        assert "1500" not in result[0]
        assert "dipinto" in result[0]

    def test_rimuove_spazi_extra(self):
        result = data_cleaner(["troppi   spazi   qui"])
        assert result == ["troppi spazi qui"]

    def test_strip(self):
        result = data_cleaner(["  spazio iniziale e finale  "])
        assert result[0] == "spazio iniziale e finale"

    def test_lista_vuota(self):
        result = data_cleaner([])
        assert result == []

    def test_testo_vuoto(self):
        result = data_cleaner([""])
        assert result == [""]

    def test_batch_multiplo(self):
        testi = ["Testo UNO!", "Testo DUE?", "Testo 3."]
        result = data_cleaner(testi)
        assert len(result) == 3
        assert result[0] == "testo uno"
        assert result[1] == "testo due"
        assert result[2] == "testo"

    def test_conserva_stopwords(self):
        """Le stopwords devono essere preservate (utili per language ID)."""
        result = data_cleaner(["the cat is on the table"])
        assert "the" in result[0]
        assert "is" in result[0]
        assert "on" in result[0]

    def test_solo_punteggiatura_e_numeri(self):
        result = data_cleaner(["123 !!!"])
        assert result == [""]


class TestPreprocessText:
    """Test per la funzione preprocess_text."""

    def test_singolo_testo(self):
        result = preprocess_text("Il Museo ha 200 opere!")
        assert result == "il museo ha opere"

    def test_coerenza_con_data_cleaner(self):
        testo = "Un testo con Numeri 42 e Punteggiatura!"
        assert preprocess_text(testo) == data_cleaner([testo])[0]
