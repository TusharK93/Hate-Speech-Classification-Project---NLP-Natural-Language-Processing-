from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import sys

from hate.pipeline.prediction_pipeline import PredictionPipeline
from hate.exception import CustomException


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="Hate Speech Detection API",
    description="NLP-based Hate Speech and Abusive Text Classification API",
    version="1.0.0"
)


# ==========================================================
# REQUEST MODEL
# ==========================================================

class TextRequest(BaseModel):

    text: str


# ==========================================================
# LOAD MODEL
# ==========================================================

prediction_pipeline = None

try:

    prediction_pipeline = PredictionPipeline()

except Exception as e:

    print(
        f"Prediction pipeline initialization failed: {e}"
    )


# ==========================================================
# HOME
# ==========================================================

@app.get(
    "/",
    tags=["Home"]
)
async def home():

    return RedirectResponse(
        url="/docs"
    )


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get(
    "/health",
    tags=["Health"]
)
async def health():

    if prediction_pipeline is None:

        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================================
# PREDICTION
# ==========================================================

@app.post(
    "/predict",
    tags=["Prediction"]
)
async def predict(request: TextRequest):

    if prediction_pipeline is None:

        return {
            "status": "error",
            "message": "Prediction model is not loaded."
        }

    try:

        result = prediction_pipeline.run_pipeline(
            request.text
        )

        return {
            "input": request.text,
            **result
        }

    except Exception as e:

        raise CustomException(
            e,
            sys
        ) from e


# ==========================================================
# LOCAL DEVELOPMENT
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )