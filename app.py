import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from hate.pipeline.prediction_pipeline import PredictionPipeline


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Hate Speech Detection API",
    description=(
        "NLP-based Hate Speech and Abusive "
        "Text Classification API"
    ),
    version="1.0.0"
)


# ============================================================
# Request Model
# ============================================================

class TextRequest(BaseModel):

    text: str


# ============================================================
# Prediction Pipeline
# ============================================================

prediction_pipeline = PredictionPipeline()


# ============================================================
# Home
# ============================================================

@app.get(
    "/",
    tags=["Home"]
)
async def home():

    return RedirectResponse(
        url="/docs"
    )


# ============================================================
# Health
# ============================================================

@app.get(
    "/health",
    tags=["Health"]
)
async def health():

    if prediction_pipeline.model_loaded:

        return {
            "status": "healthy",
            "model_loaded": True
        }


    return {
        "status": "unhealthy",
        "model_loaded": False,
        "error": getattr(
            prediction_pipeline,
            "initialization_error",
            "Unknown initialization error"
        )
    }


# ============================================================
# Prediction
# ============================================================

@app.post(
    "/predict",
    tags=["Prediction"]
)
async def predict(
    request: TextRequest
):

    result = prediction_pipeline.run_pipeline(
        request.text
    )

    return {
        "input": request.text,
        **result
    }


# ============================================================
# Local Development
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )