from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import DEFAULT_MODEL
from logger import logger
from model_loader import load_model
from routes import router


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestisce il ciclo di vita dell'applicazione:
    - All'avvio: carica il modello di default.
    - Allo spegnimento: libera la risorsa.
    """
    logger.info(f"Caricamento del modello di default: '{DEFAULT_MODEL}'")
    try:
        app.state.model = load_model(DEFAULT_MODEL)
        app.state.active_model_name = DEFAULT_MODEL
        logger.info("Modello caricato con successo.")
    except Exception as e:
        logger.error(f"Errore durante il caricamento del modello: {e}")
        app.state.model = None
        app.state.active_model_name = None

    yield  # L'applicazione è in esecuzione

    # Cleanup all'arresto
    app.state.model = None
    app.state.active_model_name = None
    logger.info("Modello rilasciato. Applicazione terminata.")


# ---------------------------------------------------------------------------
# Inizializzazione dell'Applicazione FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="MuseumLangAPI",
              description="API REST per il riconoscimento automatico della lingua di testi museali. Supporta italiano (IT), inglese (EN) e tedesco (DE).",
              lifespan=lifespan)

app.include_router(router)
