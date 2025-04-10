import argparse
import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
import logging
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
logger.info(f"GPU available: {len(gpus) > 0}, Devices: {gpus}")

class MLflowLoggingCallback(Callback):
    def __init__(self, model_name, save_dir="../ml/models"):
        super().__init__()
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Log metrics to MLflow
        metrics = {
            "train_loss": logs.get('loss'),
            "train_mae": logs.get('mae'),
            "train_mse": logs.get('mse'),
            "val_loss": logs.get('val_loss'),
            "val_mae": logs.get('val_mae'),
            "val_mse": logs.get('val_mse')
        }
        for name, value in metrics.items():
            if value is not None:
                mlflow.log_metric(name, value, step=epoch)

        # Save model checkpoint
        model_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch+1}.keras")
        self.model.save(model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

def load_and_preprocess_data(csv_path, features, window_size):
    try:
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['time'])
        df = df.sort_values('time').set_index('time')
        df = df[features]

        original_energy = df['energy'].copy()
        energy = df['energy'].copy()
        mask = energy == 0
        
        # Keep initial zeros until first non-zero
        first_nonzero_idx = energy.ne(0).idxmax()
        mask.loc[:first_nonzero_idx] = False
        
        # Handle subsequent zeros
        if mask.any():
            energy.loc[mask] = np.nan
            energy = energy.ffill()
        df['energy'] = energy.fillna(method='ffill')

        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i - window_size:i])
            y.append(scaled_data[i])

        return np.array(X), np.array(y), scaler, df.index[window_size:], original_energy
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def build_model(input_shape, hidden_size, num_layers, output_size, learning_rate):
    logger.info("Building LSTM model")
    model = Sequential([
        LSTM(hidden_size, return_sequences=(num_layers > 1), input_shape=input_shape),
        *[LSTM(hidden_size, return_sequences=(i < num_layers - 1)) for i in range(1, num_layers)],
        Dense(hidden_size, activation='relu'),
        Dense(output_size)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def plot_metrics(history, save_path="../ml/plots"):
    os.makedirs(save_path, exist_ok=True)
    
    metrics = ['loss', 'mae', 'mse']
    for metric in metrics:
        if metric in history.history:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[metric], label=f'Train {metric.upper()}')
            plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.upper()}')
            plt.title(f'{metric.upper()} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            plt_path = os.path.join(save_path, f"{metric}_plot.png")
            plt.savefig(plt_path)
            plt.close()
            mlflow.log_artifact(plt_path)

def main(args):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(vars(args))

        # Data preprocessing
        features = ['energy']
        X, y, scaler, time_index, original_energy = load_and_preprocess_data(
            args.csv_path, features, args.window_size
        )

        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Build and train model
        model = build_model(
            (args.window_size, len(features)),
            args.hidden_size,
            args.num_layers,
            len(features),
            args.learning_rate
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1),
            MLflowLoggingCallback(model_name=args.model_name)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks
        )

        # Evaluation
        metrics = model.evaluate(X_test, y_test, return_dict=True)
        mlflow.log_metrics(metrics)

        # Save final model
        final_path = Path(f"../ml/models/{args.model_name}.keras")
        model.save(final_path)
        mlflow.log_artifact(str(final_path))

        # Plotting
        plot_metrics(history)
        
        # Predictions and visualization
        y_pred = model.predict(X_test)
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_index[split_idx:], y_test_inv[:, 0], label='Actual')
        plt.plot(time_index[split_idx:], y_pred_inv[:, 0], label='Predicted')
        plt.title('Energy Prediction vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        pred_plot = "../ml/plots/predictions.png"
        plt.savefig(pred_plot)
        plt.close()
        mlflow.log_artifact(pred_plot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM model for energy prediction")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--window_size', type=int, default=672, help='Input window size')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_name', type=str, default="lstm_energy")
    
    args = parser.parse_args()
    main(args)