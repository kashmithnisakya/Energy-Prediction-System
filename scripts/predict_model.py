import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(csv_path, features, window_size, scaler=None):
    """
    Load and preprocess data for prediction, consistent with training preprocessing
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['time'])
    df = df.sort_values('time').set_index('time')
    df = df[features]

    # Handle zero values in energy (keeping initial zeros)
    logger.info("Processing zero values in energy")
    energy = df['energy'].copy()
    mask = energy == 0
    
    # Keep initial zeros until first non-zero value
    first_nonzero_idx = energy.ne(0).idxmax()
    mask.loc[:first_nonzero_idx] = False
    
    # Replace subsequent zeros with previous non-zero value
    if mask.any():
        energy.loc[mask] = np.nan
        energy = energy.fillna(method='ffill')
    df['energy'] = energy
    
    # Fill any remaining NaN values
    df = df.fillna(method='ffill')

    # Scale data using provided scaler or create new one if not provided
    if scaler is None:
        logger.info("No scaler provided, creating new one")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
    else:
        logger.info("Using provided scaler")
        scaled_data = scaler.transform(df)

    # Create sequences
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
    
    X = np.array(X)
    logger.info(f"Created prediction sequences with shape: {X.shape}")
    return X, df.index[window_size:], scaler

def load_trained_model(model_path):
    """Load the trained LSTM model"""
    logger.info(f"Loading model from {model_path}")
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict(model, X, scaler, features):
    """Make predictions using the loaded model"""
    logger.info("Making predictions")
    predictions = model.predict(X)
    
    # Inverse transform predictions
    logger.info("Inverse transforming predictions")
    predictions_inv = scaler.inverse_transform(predictions)
    return predictions_inv

def save_predictions(predictions, timestamps, features, output_path):
    """Save predictions to CSV"""
    logger.info(f"Saving predictions to {output_path}")
    pred_df = pd.DataFrame(
        predictions,
        columns=features,
        index=timestamps
    )
    pred_df.index.name = 'time'
    pred_df.to_csv(output_path)
    logger.info("Predictions saved successfully")

def main(args):
    # Define features consistent with training
    features = ['energy', 'power']

    # Load the trained model
    model = load_trained_model(args.model_path)

    # Load and preprocess prediction data
    # Note: In a real scenario, you might want to pass the training scaler
    # Here we're creating a new one for simplicity
    X, timestamps, scaler = load_and_preprocess_data(
        args.csv_path,
        features,
        args.window_size
    )

    # Make predictions
    predictions = predict(model, X, scaler, features)

    # Save predictions
    save_predictions(predictions, timestamps, features, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions using trained LSTM model")
    parser.add_argument('--csv_path', type=str, required=True, 
                       help='Path to CSV file with input data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained LSTM model (.h5 file)')
    parser.add_argument('--window_size', type=int, default=672,
                       help='Input window size (should match training)')
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                       help='Path to save prediction results')

    args = parser.parse_args()
    main(args)
