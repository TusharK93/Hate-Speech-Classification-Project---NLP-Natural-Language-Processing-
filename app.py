from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from hate.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI(
    title="Hate Speech Detection API",
    version="1.0.0",
    description="NLP-based Hate Speech and Abusive Text Classification API"
)


class TextRequest(BaseModel):
    text: str


# ==========================================================
# LOAD MODEL ONCE
# ==========================================================

try:

    prediction_pipeline = PredictionPipeline()

    MODEL_LOADED = True

except Exception as e:

    prediction_pipeline = None
    MODEL_LOADED = False

    print(
        f"Prediction pipeline initialization failed: {e}"
    )


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
async def home():

    return RedirectResponse(
        url="/docs"
    )


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED
    }


# ==========================================================
# PREDICTION
# ==========================================================

@app.post("/predict")
async def predict(request: TextRequest):

    if not MODEL_LOADED:

        return {
            "status": "error",
            "message": "Prediction model is not loaded."
        }

    try:

        result = prediction_pipeline.run_pipeline(
            request.text
        )

        return {
            "status": "success",
            "input": request.text,
            **result
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ==========================================================
# LOCAL
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )