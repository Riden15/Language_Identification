import logging
import os
import pickle
import urllib.request
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configurazione del Logging
# ---------------------------------------------------------------------------
# Il logger scrive sia su file (con rotazione automatica) che sulla console,

LOG_FILE = "api.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file di log
LOG_BACKUP_COUNT = 3  # mantieni al massimo 3 file di backup

log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("MuseumLangApi")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    # Handler su console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Handler su file con rotazione automatica
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Caricamento del Modello
# ---------------------------------------------------------------------------

MODEL_URL = "https://raw.githubusercontent.com/Profession-AI/progetti-python/refs/heads/main/Messa%20in%20produzione%20di%20un%20sistema%20per%20il%20riconoscimento%20della%20lingua%20di%20testi%20per%20un%20museo/language_detection_pipeline.pkl"
MODEL_PATH = os.path.join(os.getcwd(), "../models/language_detection_pipeline.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestisce il ciclo di vita dell'applicazione:
    - All'avvio: carica il modello dal file pickle e lo salva in app.state.pipeline.
    - Allo spegnimento: libera la risorsa.
    """

    # Download del modello se non già presente
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Modello non trovato localmente. Download da: {MODEL_URL}")
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info(f"Modello scaricato e salvato in: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Errore durante il download del modello: {e}")
            app.state.pipeline = None
            yield
            return

    # Caricamento del modello
    logger.info(f"Caricamento del modello da: {MODEL_PATH}")
    try:
        with open(MODEL_PATH, "rb") as f:
            app.state.pipeline = pickle.load(f)
        logger.info("Modello caricato con successo.")
    except Exception as e:
        logger.error(f"Errore durante il caricamento del modello: {e}")
        app.state.pipeline = None

    yield  # L'applicazione è in esecuzione

    # Cleanup all'arresto
    app.state.pipeline = None
    logger.info("Modello rilasciato. Applicazione terminata.")


# ---------------------------------------------------------------------------
# Inizializzazione dell'Applicazione FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="MuseumLangAPI",
              description="API REST per il riconoscimento automatico della lingua di testi museali. Supporta italiano (IT), inglese (EN) e tedesco (DE).",
              lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemi Pydantic (validazione input/output)
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    """
    Schema di input per l'endpoint di identificazione lingua
    """
    model_config = {
        "extra": "forbid",  # Non accettare campi extra
        "json_schema_extra": {
            "example": {"text": "Questo è un esempio di testo in italiano."}
        }
    }

    text: str = Field(..., min_length=1, description="Il testo da analizzare (non può essere vuoto).")


class LanguageResponse(BaseModel):
    """
    Schema di output per la risposta dell'endpoint di identificazione lingua.
    Contiene un dizionario con le probabilità per ciascuna lingua e la lingua predetta.
    """
    predicted_probability: Dict[str, float] = Field(..., description='Predicted prob')
    predicted_cls: str = Field(...)


class ErrorResponse(BaseModel):
    """
    Schema di output per i messaggi di errore
    """
    error: str


# ---------------------------------------------------------------------------
# Endpoint: POST /identify-language
# ---------------------------------------------------------------------------

@app.post("/identify-language",
          response_model=LanguageResponse,
          summary="Identifica la lingua di un testo",
          description="Riceve un testo in input e restituisce la lingua identificata (IT, EN, DE) con le relative probabilità.")
def identify_language(payload: TextInput) -> LanguageResponse:
    """
    Identifica la lingua del testo fornito.
    """

    # Verifica disponibilità modello
    language_identification_model = app.state.pipeline
    if language_identification_model is None:
        logger.error("Richiesta ricevuta ma il modello non è disponibile.")
        raise HTTPException(status_code=503, detail="Il modello di riconoscimento lingua non è disponibile.")

    logger.info(
        f"Richiesta ricevuta | testo: \"{payload.text[:100] + "..." if len(payload.text) > 100 else payload.text}\"")

    # Previsione
    try:
        predicted_language = language_identification_model.predict([payload.text])[0]
        predicted_proba = language_identification_model.predict_proba([payload.text])[0]
        target_names = language_identification_model.classes_
        predicted_proba_dict = {target_names[i]: predicted_proba[i] for i in range(len(predicted_proba))}

    except Exception as e:
        logger.error(f"Errore durante la previsione: {e}")
        raise HTTPException(status_code=500,
                            detail="Si è verificato un errore interno durante il riconoscimento della lingua.")

    # Log della risposta
    logger.info(f"Risposta inviata | language_code: {predicted_language} | confidence: {predicted_proba_dict}")

    return LanguageResponse(predicted_probability=predicted_proba_dict, predicted_cls=predicted_language)


# ---------------------------------------------------------------------------
# Endpoint: POST /predict-file
# ---------------------------------------------------------------------------

@app.post("/predict-file",
          response_model=List[LanguageResponse],
          summary="Identifica la lingua di più testi da file",
          description="Riceve un file .txt con un testo per riga e restituisce una lista di oggetti con la lingua identificata e le probabilità per ciascuna riga."
          )
def predict_file(input_file: UploadFile) -> List[LanguageResponse]:
    """
    Identifica la lingua di ogni riga del file caricato.
    """

    # Verifica disponibilità modello
    pipeline = app.state.pipeline
    if pipeline is None:
        logger.error("Richiesta ricevuta ma il modello non è disponibile.")
        raise HTTPException(status_code=503, detail="Il modello di riconoscimento lingua non è disponibile.")

    # Lettura del file
    try:
        content = input_file.file.read().decode("utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
    except Exception as e:
        logger.error(f"Errore durante la lettura del file: {e}")
        raise HTTPException(status_code=400, detail="Impossibile leggere il file caricato.")

    if not lines:
        raise HTTPException(status_code=400, detail="Il file è vuoto o non contiene righe valide.")

    logger.info(f"Richiesta /predict-file ricevuta | righe da processare: {len(lines)}")

    # Previsione
    try:
        predicted_languages = pipeline.predict(lines)
        predicted_probability = pipeline.predict_proba(lines)
        target_names = pipeline.classes_

        output = []
        for pred_cls, pred_probability_row in zip(predicted_languages, predicted_probability):
            proba_dict = {target_names[i]: float(pred_probability_row[i]) for i in range(len(pred_probability_row))}
            output.append(LanguageResponse(predicted_probability=proba_dict, predicted_cls=pred_cls))

    except Exception as e:
        logger.error(f"Errore durante la previsione batch: {e}")
        raise HTTPException(status_code=500, detail="Si è verificato un errore durante il riconoscimento della lingua.")

    logger.info(f"Risposta /predict-file inviata | righe processate: {len(output)}")
    return output


# ---------------------------------------------------------------------------
# Entry point per l'avvio diretto con 'python app.py'
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app")
