import logging
import os
import pickle
import urllib.request
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Configurazione del Logging
# ---------------------------------------------------------------------------
# Il logger scrive sia su file (con rotazione automatica) che sulla console,

LOG_FILE = "api.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file di log
LOG_BACKUP_COUNT = 3  # mantieni al massimo 3 file di backup

# Formato del log: timestamp | livello | messaggio
log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("MuseumLangApi")
logger.setLevel(logging.INFO)

# Handler su file con rotazione automatica
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Caricamento del Modello
# ---------------------------------------------------------------------------
# Il modello viene caricato una sola volta all'avvio dell'applicazione tramite
# il meccanismo 'lifespan' di FastAPI. Viene salvato in un dizionario condiviso
# (app.state) accessibile da tutti gli endpoint.

MODEL_URL = "https://github.com/Profession-AI/progetti-python/raw/refs/heads/main/Messa%20in%20produzione%20di%20un%20sistema%20per%20il%20riconoscimento%20della%20lingua%20di%20testi%20per%20un%20museo/languagedetectionpipeline.pkl"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "languagedetectionpipeline.pkl")


# ---------------------------------------------------------------------------
# Lifespan: download e caricamento del modello all'avvio
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Scarica (se necessario) e carica il modello all'avvio; lo libera allo shutdown."""
    # --- Download del modello ---
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Modello non trovato in locale. Download da:\n  {MODEL_URL}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info(f"Modello scaricato con successo in: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Errore durante il download del modello: {e}")
            app.state.pipeline = None
            yield
            return
    else:
        logger.info(f"Modello già presente in locale: {MODEL_PATH}")

    # --- Caricamento del modello ---
    try:
        with open(MODEL_PATH, "rb") as f:
            app.state.pipeline = pickle.load(f)
        logger.info("Pipeline caricata con successo.")
    except Exception as e:
        logger.error(f"Errore durante il caricamento del modello: {e}")
        app.state.pipeline = None

    yield  # L'applicazione è in esecuzione

    # --- Cleanup allo shutdown ---
    app.state.pipeline = None
    logger.info("Pipeline rimossa dalla memoria.")


# ---------------------------------------------------------------------------
# Inizializzazione dell'Applicazione FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MuseumLangAPI",
    description=(
        "API REST per il riconoscimento automatico della lingua di testi museali. "
        "Supporta italiano (IT), inglese (EN) e tedesco (DE)."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemi Pydantic (validazione input/output)
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    """Schema di input per l'endpoint di identificazione lingua."""
    text: str
    model_config = {
        "json_schema_extra": {
            "example": {"text": "Questo è un esempio di testo in italiano."}
        }
    }


class LanguageResponse(BaseModel):
    """Schema di output con il codice lingua identificato e la confidenza."""
    language_code: str
    confidence: float


class ErrorResponse(BaseModel):
    """Schema di output per i messaggi di errore."""
    error: str


# ---------------------------------------------------------------------------
# Funzione ausiliaria: estrazione confidenza
# ---------------------------------------------------------------------------

def get_confidence(pipeline: Any, text: str) -> float:
    """
    Calcola la confidenza della previsione.

    Tenta di usare predict_proba() (supportato da Naive Bayes, MLP, ecc.).
    Se il classificatore non supporta predict_proba (es. LinearSVC), restituisce 1.0
    come valore di fallback conservativo.

    Args:
        pipeline: La sklearn Pipeline caricata dal pickle.
        text:     Il testo di cui si vuole la confidenza.

    Returns:
        Confidenza come float tra 0.0 e 1.0.
    """
    try:
        probabilities = pipeline.predict_proba([text])
        # predict_proba restituisce un array (n_samples, n_classes); prendiamo il max
        confidence = float(np.max(probabilities))
    except AttributeError:
        # Il classificatore non supporta predict_proba (es. LinearSVC)
        logger.warning(
            "Il classificatore non supporta predict_proba(). "
            "Confidenza impostata a 1.0 come valore di fallback."
        )
        confidence = 1.0
    return round(confidence, 4)


# ---------------------------------------------------------------------------
# Endpoint: POST /identify-language
# ---------------------------------------------------------------------------

@app.post(
    "/identify-language",
    response_model=LanguageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Testo vuoto o non valido."},
        503: {"model": ErrorResponse, "description": "Modello non disponibile."},
        500: {"model": ErrorResponse, "description": "Errore interno del server."},
    },
    summary="Identifica la lingua di un testo",
    description=(
            "Riceve un testo in formato JSON e restituisce il codice ISO della lingua "
            "riconosciuta (IT, EN, DE) insieme alla confidenza della previsione."
    ),
)
async def identify_language(payload: TextInput) -> LanguageResponse:
    """
    Identifica la lingua del testo fornito.

    - **text**: Il testo da analizzare (non può essere vuoto).

    Restituisce:
    - **language_code**: Codice ISO della lingua (es. "IT", "EN", "DE").
    - **confidence**: Probabilità associata alla previsione (0.0 – 1.0).
    """
    # --- Verifica disponibilità modello ---
    pipeline = app.state.pipeline
    if pipeline is None:
        logger.error("Richiesta ricevuta ma il modello non è disponibile.")
        raise HTTPException(
            status_code=503,
            detail="Il modello di riconoscimento lingua non è disponibile. "
                   "Contattare l'amministratore del sistema.",
        )

    # --- Log della richiesta ---
    # Il testo viene troncato nel log per evitare entry troppo lunghe
    text_preview = payload.text[:100] + "..." if len(payload.text) > 100 else payload.text
    logger.info(f"Richiesta ricevuta | testo: \"{text_preview}\"")

    # --- Previsione ---
    try:
        predicted_language = pipeline.predict([payload.text])[0]
        confidence = get_confidence(pipeline, payload.text)
    except Exception as e:
        logger.error(f"Errore durante la previsione: {e}")
        raise HTTPException(
            status_code=500,
            detail="Si è verificato un errore interno durante il riconoscimento della lingua.",
        )

    # --- Log della risposta ---
    logger.info(
        f"Risposta inviata    | language_code: {predicted_language} | confidence: {confidence}"
    )

    return LanguageResponse(language_code=predicted_language, confidence=confidence)


# ---------------------------------------------------------------------------
# Entry point per l'avvio diretto con 'python app.py'
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        reload=False,  # reload=True utile solo in sviluppo
        log_level="info",
    )
