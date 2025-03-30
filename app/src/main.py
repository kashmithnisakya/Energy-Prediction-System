from fastapi import FastAPI
from fastapi.responses import FileResponse
from src.config import config
from src.models import PredictionRequest
from src.services.data_processing import load_and_preprocess_data
from src.services.prediction import prediction_service
from src.utils.logging import logger
from pathlib import Path

app = FastAPI(
    title="LSTM Prediction API",
    description="API for power and energy prediction using LSTM",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    prediction_service.load_model(config.MODEL_PATH)

@app.post("/predict/", response_class=FileResponse)
async def predict_file(request: PredictionRequest):
    """
    Make predictions using a CSV file path and factory name.
    Returns predictions as a downloadable CSV file.
    Expects a CSV with 'time', 'energy', and 'power' columns.
    """
    csv_path = Path(request.csv_path)
    factory_name = request.factory_name

    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {csv_path}")

    try:
        # Preprocess data
        X, timestamps, scaler = load_and_preprocess_data(csv_path, config.FEATURES, config.WINDOW_SIZE)

        # Make predictions
        predictions = prediction_service.predict(X, scaler, config.FEATURES)

        # Save predictions
        temp_output_path = config.TEMP_DIR / f"predictions_{factory_name}_{csv_path.name}"
        prediction_service.save_predictions(predictions, timestamps, config.FEATURES, temp_output_path, factory_name)

        return FileResponse(
            path=temp_output_path,
            filename=f"predictions_{factory_name}_{csv_path.name}",
            media_type='text/csv'
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": prediction_service.model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)