from typing import List

from fastapi import APIRouter, HTTPException, Request, UploadFile
from config import AVAILABLE_MODELS
from utils import load_model, preprocess_text, data_cleaner
from logger import logger
from schemas import LanguageResponse, ModelInfoResponse, ModelSwitchInput, TextInput

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoint: GET /model-info
# ---------------------------------------------------------------------------

@router.get("/model-info",
            response_model=ModelInfoResponse,
            summary="Informazioni sul modello attivo",
            description="Restituisce il nome del modello attualmente attivo e la lista dei modelli disponibili.")
def model_info(request: Request) -> ModelInfoResponse:
    return ModelInfoResponse(
        active_model=request.app.state.active_model_name or "nessuno",
        available_models=list(AVAILABLE_MODELS.keys()),
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /switch-model
# ---------------------------------------------------------------------------

@router.post("/switch-model",
             response_model=ModelInfoResponse,
             summary="Cambia il modello attivo",
             description="Permette di cambiare il modello di riconoscimento lingua a runtime.")
def switch_model(payload: ModelSwitchInput, request: Request) -> ModelInfoResponse:
    model_name = payload.model_name

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400,
                            detail=f"Modello '{model_name}' non disponibile. Modelli ammessi: {list(AVAILABLE_MODELS.keys())}")

    if model_name == request.app.state.active_model_name:
        logger.info(f"Modello '{model_name}' già attivo, nessun cambio necessario.")
        return ModelInfoResponse(active_model=model_name, available_models=list(AVAILABLE_MODELS.keys()))

    logger.info(f"Cambio modello: '{request.app.state.active_model_name}' -> '{model_name}'")
    try:
        request.app.state.model = load_model(model_name)
        request.app.state.active_model_name = model_name
        logger.info(f"Modello '{model_name}' caricato con successo.")
    except Exception as e:
        logger.error(f"Errore durante il cambio modello: {e}")
        raise HTTPException(status_code=500, detail=f"Errore durante il caricamento del modello '{model_name}'.")

    return ModelInfoResponse(active_model=model_name, available_models=list(AVAILABLE_MODELS.keys()))


# ---------------------------------------------------------------------------
# Endpoint: POST /identify-language
# ---------------------------------------------------------------------------

@router.post("/identify-language",
             response_model=LanguageResponse,
             summary="Identifica la lingua di un testo",
             description="Riceve un testo in input e restituisce la lingua identificata (IT, EN, DE) con le relative probabilità.")
def identify_language(payload: TextInput, request: Request) -> LanguageResponse:
    """
    Identifica la lingua del testo fornito.
    """

    # Verifica disponibilità modello e vectorizer
    language_identification_model = request.app.state.model
    vectorizer = request.app.state.vectorizer
    if language_identification_model is None or vectorizer is None:
        logger.error("Richiesta ricevuta ma il modello o il vectorizer non è disponibile.")
        raise HTTPException(status_code=503, detail="Il modello di riconoscimento lingua non è disponibile.")

    logger.info(
        f"Richiesta ricevuta | testo: \"{payload.text[:100] + '...' if len(payload.text) > 100 else payload.text}\"")

    # Preprocessing e Previsione
    try:
        cleaned_text = preprocess_text(payload.text)
        text_to_predict = vectorizer.transform([cleaned_text])

        predicted_language = language_identification_model.predict(text_to_predict)[0]
        predicted_proba = language_identification_model.predict_proba(text_to_predict)[0]
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

@router.post("/predict-file",
             response_model=List[LanguageResponse],
             summary="Identifica la lingua di più testi da file",
             description="Riceve un file .txt con un testo per riga e restituisce una lista di oggetti con la lingua identificata e le probabilità per ciascuna riga."
             )
def predict_file(input_file: UploadFile, request: Request) -> List[LanguageResponse]:
    """
    Identifica la lingua di ogni riga del file caricato.
    """

    # Verifica disponibilità modello e vectorizer
    language_identification_model = request.app.state.model
    vectorizer = request.app.state.vectorizer
    if language_identification_model is None or vectorizer is None:
        logger.error("Richiesta ricevuta ma il modello o il vectorizer non è disponibile.")
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

    # Preprocessing e Previsione
    try:
        lines = data_cleaner(lines)
        lines_tfidf = vectorizer.transform(lines)
        predicted_languages = language_identification_model.predict(lines_tfidf)
        predicted_probability = language_identification_model.predict_proba(lines_tfidf)
        target_names = language_identification_model.classes_

        output = []
        for pred_cls, pred_probability_row in zip(predicted_languages, predicted_probability):
            proba_dict = {target_names[i]: float(pred_probability_row[i]) for i in range(len(pred_probability_row))}
            output.append(LanguageResponse(predicted_probability=proba_dict, predicted_cls=pred_cls))

    except Exception as e:
        logger.error(f"Errore durante la previsione batch: {e}")
        raise HTTPException(status_code=500, detail="Si è verificato un errore durante il riconoscimento della lingua.")

    logger.info(f"Risposta /predict-file inviata | righe processate: {len(output)}")
    return output
