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
    """Load and preprocess data, consistent with training preprocessing"""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['time'])
    df = df.sort_values('time').set_index('time')
    df = df[features]

    # Handle zero values in energy (keeping initial zeros)
    logger.info("Processing zero values in energy")
    energy = df['energy'].copy()
    mask = energy == 0
    first_nonzero_idx = energy.ne(0).idxmax()
    mask.loc[:first_nonzero_idx] = False
    if mask.any():
        energy.loc[mask] = np.nan
        energy = energy.fillna(method='ffill')
    df['energy'] = energy
    
    df = df.fillna(method='ffill')

    # Scale data
    if scaler is None:
        logger.info("No scaler provided, creating new one")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
    else:
        logger.info("Using provided scaler")
        scaled_data = scaler.transform(df)

    logger.info(f"Processed data with shape: {scaled_data.shape}")
    return scaled_data, df.index, scaler

def load_trained_model(model_path):
    """Load the trained LSTM model"""
    logger.info(f"Loading model from {model_path}")
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_future(model, initial_sequence, scaler, features, future_steps):
    """Predict future timesteps iteratively"""
    logger.info(f"Predicting {future_steps} future steps")
    predictions = []
    current_sequence = initial_sequence.copy()  # Shape: (window_size, n_features)

    for _ in range(future_steps):
        # Reshape for LSTM input: (1, window_size, n_features)
        X = current_sequence.reshape((1, current_sequence.shape[0], current_sequence.shape[1]))
        # Predict next timestep
        next_pred = model.predict(X, verbose=0)
        # Append to predictions
        predictions.append(next_pred[0])
        # Update sequence: remove oldest timestep, add new prediction
        current_sequence = np.vstack((current_sequence[1:], next_pred[0]))

    # Convert predictions to array and inverse transform
    predictions = np.array(predictions)
    predictions_inv = scaler.inverse_transform(predictions)
    return predictions_inv

def save_predictions(predictions, last_timestamp, features, output_path, freq='1H'):
    """Save future predictions to CSV with appropriate timestamps"""
    logger.info(f"Saving predictions to {output_path}")
    # Generate future timestamps (assuming hourly data by default, adjust freq as needed)
    future_timestamps = pd.date_range(
        start=last_timestamp, 
        periods=len(predictions) + 1, 
        freq=freq
    )[1:]  # Skip the starting timestamp
    pred_df = pd.DataFrame(
        predictions,
        columns=features,
        index=future_timestamps
    )
    pred_df.index.name = 'time'
    pred_df.to_csv(output_path)
    logger.info("Predictions saved successfully")

def main(args):
    # Define features consistent with training
    features = ['energy', 'power']

    # Load the trained model
    model = load_trained_model(args.model_path)

    # Load and preprocess data
    scaled_data, timestamps, scaler = load_and_preprocess_data(
        args.csv_path,
        features,
        args.window_size
    )

    # Use the last window_size timesteps as the initial sequence
    if len(scaled_data) < args.window_size:
        raise ValueError(f"Input data has {len(scaled_data)} rows, but window_size is {args.window_size}")
    initial_sequence = scaled_data[-args.window_size:]

    # Predict future steps
    predictions = predict_future(
        model, 
        initial_sequence, 
        scaler, 
        features, 
        args.future_steps
    )

    # Save predictions with future timestamps
    last_timestamp = timestamps[-1]
    save_predictions(predictions, last_timestamp, features, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict future values using trained LSTM model")
    parser.add_argument('--csv_path', type=str, required=True, 
                        help='Path to CSV file with historical data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained LSTM model (.h5 file)')
    parser.add_argument('--window_size', type=int, default=672,
                        help='Input window size (must match training)')
    parser.add_argument('--future_steps', type=int, required=True,
                        help='Number of future timesteps to predict')
    parser.add_argument('--output_path', type=str, default='future_predictions.csv',
                        help='Path to save prediction results')

    args = parser.parse_args()
    main(args)