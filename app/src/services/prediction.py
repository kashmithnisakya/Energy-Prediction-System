import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from fastapi import HTTPException
from src.config import config
from src.utils.logging import logger

class PredictionService:
    def __init__(self):
        self.model = None

    def load_model(self, model_path: Path):
        """Load the trained LSTM model"""
        logger.info(f"Loading model from {model_path}")
        try:
            self.model = load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    def predict(self, X, scaler, features):
        """Make predictions using the loaded model"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        logger.info("Making predictions")
        predictions = self.model.predict(X)
        predictions_inv = scaler.inverse_transform(predictions)
        return predictions_inv

    def save_predictions(self, predictions, timestamps, features, output_path: Path, factory_name: str):
        """Save predictions to CSV"""
        logger.info(f"Saving predictions to {output_path}")
        pred_df = pd.DataFrame(predictions, columns=features, index=timestamps)
        pred_df.index.name = 'time'
        pred_df['factory_name'] = factory_name
        pred_df.to_csv(output_path)

prediction_service = PredictionService()