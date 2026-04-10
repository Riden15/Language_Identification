# 🌐 Language Identification — MuseumLangID

API REST per il riconoscimento automatico della lingua di testi museali. Supporta **italiano (IT)**, **inglese (EN)** e*
*tedesco (DE)**.

Un museo internazionale necessita di identificare automaticamente la lingua delle descrizioni delle opere d'arte scritte
in più lingue. Il progetto sviluppa un sistema di Machine Learning per l'identificazione automatica della lingua,
esposto tramite API REST con FastAPI.

## Struttura del progetto

```
├── images/                  # Grafici e visualizzazioni generate dal notebook
├── logs/                    # File di log dell'API (creati a runtime)
├── models/                  # Modelli serializzati (.pkl), generati dal notebook
├── src/
│   ├── config.py            # Costanti di configurazione (logging, modelli)
│   ├── logger.py            # Setup del logging (console + file con rotazione)
│   ├── model_loader.py      # Utility per il caricamento dei modelli pickle
│   ├── schemas.py           # Schemi Pydantic per validazione input/output
│   ├── routes.py            # Definizione degli endpoint API
│   ├── MuseumLangAPI.py     # Creazione app FastAPI e gestione lifespan
│   └── main.py              # Entry point per l'avvio del server
├── Language Identification - Models study.ipynb  # Studio e addestramento dei modelli
├── requirements.txt
└── README.md
```

## Modelli

Nel notebook vengono addestrati e confrontati 3 modelli di classificazione, tutti basati su feature TF-IDF (unigrammi +
bigrammi):

| Modello                         | Accuracy | Precision | Recall | F1-Score |
|---------------------------------|----------|-----------|--------|----------|
| **Naive Bayes (MultinomialNB)** | 1.00     | 1.00      | 1.00   | 1.00     |
| **SVM (LinearSVC)**             | 1.00     | 1.00      | 1.00   | 1.00     |
| **MLP (256 → 128, ReLU, Adam)** | 1.00     | 1.00      | 1.00   | 1.00     |

Tutti e tre i modelli raggiungono performance perfette. Per la produzione si consiglia **Naive Bayes** o **Linear SVC**
per la maggiore velocità di training e inferenza e l'assenza di iperparametri critici.

> **Nota:** I modelli non sono inclusi nel repository. È necessario eseguire il notebook
`Language Identification - Models study.ipynb` prima di avviare l'API, per generare i file `.pkl` nella cartella
`models/`.

## Prerequisiti

- Python 3.10+
- Le dipendenze elencate in `requirements.txt`

## Installazione

```bash
pip install -r requirements.txt
```

## Generazione dei modelli

Prima di avviare l'API, eseguire tutte le celle del notebook:

```
src/Language Identification - Models study.ipynb
```

Questo creerà i seguenti file nella cartella `models/`:

- `naive_bayes_model.pkl`
- `svm_model.pkl`
- `mlp_model.pkl`

## Avvio dell'API

```bash
cd src
python main.py
```

Il server sarà disponibile su `http://127.0.0.1:8000`. La documentazione interattiva è accessibile su
`http://127.0.0.1:8000/docs`.

## Endpoint

### `GET /model-info`

Restituisce il modello attualmente attivo e la lista dei modelli disponibili.

**Risposta:**

```json
{
  "active_model": "naive_bayes",
  "available_models": [
    "naive_bayes",
    "svm",
    "mlp"
  ]
}
```

### `POST /switch-model`

Cambia il modello attivo a runtime.

**Body:**

```json
{
  "model_name": "svm"
}
```

### `POST /identify-language`

Identifica la lingua di un singolo testo.

**Body:**

```json
{
  "text": "Questo è un esempio di testo in italiano."
}
```

**Risposta:**

```json
{
  "predicted_probability": {
    "DE": 0.001,
    "EN": 0.002,
    "IT": 0.997
  },
  "predicted_cls": "IT"
}
```

### `POST /predict-file`

Identifica la lingua di più testi caricando un file `.txt` (un testo per riga).

**Body:** `multipart/form-data` con campo `input_file`.
