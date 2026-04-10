from typing import Dict, List

from pydantic import BaseModel, Field

from config import AVAILABLE_MODELS


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


class ModelSwitchInput(BaseModel):
    """
    Schema di input per cambiare il modello attivo.
    """
    model_config = {"extra": "forbid"}

    model_name: str = Field(
        ...,
        description=f"Nome del modello da attivare. Valori ammessi: {', '.join(AVAILABLE_MODELS.keys())}",
    )


class ModelInfoResponse(BaseModel):
    """
    Schema di output con le informazioni sul modello attivo.
    """
    active_model: str = Field(..., description="Nome del modello attualmente attivo.")
    available_models: List[str] = Field(..., description="Lista dei modelli disponibili.")


class ErrorResponse(BaseModel):
    """
    Schema di output per i messaggi di errore
    """
    error: str
