import argparse
import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
import logging
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    logger.info(f"GPU is available and will be used: {gpus}")
else:
    logger.info("No GPU detected, using CPU")

# Custom callback for MLflow logging per epoch
class MLflowLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mlflow.log_metric("train_loss", logs.get('loss'), step=epoch)
        mlflow.log_metric("train_mae", logs.get('mae'), step=epoch)
        mlflow.log_metric("val_loss", logs.get('val_loss'), step=epoch)
        mlflow.log_metric("val_mae", logs.get('val_mae'), step=epoch)

def load_and_preprocess_data(csv_path, features, window_size):
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

    logger.info("Scaling data")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])

    logger.info("Creating input sequences")
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def build_model(input_shape, hidden_size, num_layers, output_size, learning_rate):
    logger.info("Building LSTM model")
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    for _ in range(1, num_layers):
        model.add(LSTM(hidden_size, return_sequences=(_ < num_layers - 1)))
    model.add(Dense(output_size))
    
    # Custom optimizer with specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def main(args):
    logger.info("Starting MLflow run")
    mlflow.start_run()

    # Log all parameters including learning rate
    params = vars(args)
    mlflow.log_params(params)

    features = ['energy', 'power']
    X, y, scaler = load_and_preprocess_data(args.csv_path, features, args.window_size)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Testing data shapebeer: {X_test.shape}, {y_test.shape}")

    # Learning rate scheduler
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )

    model = build_model(
        (args.window_size, len(features)),
        args.hidden_size,
        args.num_layers,
        len(features),
        args.learning_rate
    )
    model.summary(print_fn=logger.info)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    mlflow_callback = MLflowLoggingCallback()

    logger.info("Starting model training")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, lr_schedule, mlflow_callback]
    )

    logger.info("Evaluating model")
    loss, mae = model.evaluate(X_test, y_test)
    mlflow.log_metrics({"final_loss": loss, "final_mae": mae})

    model_path = Path("../ml/model/lstm_model.h5")
    logger.info(f"Saving model to {model_path}")
    model.save(model_path)
    mlflow.log_artifact(str(model_path))

    logger.info("MLflow run completed")
    mlflow.end_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM model for power and energy prediction")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to processed CSV file')
    parser.add_argument('--window_size', type=int, default=672, help='Input window size (e.g. 14 days)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')

    args = parser.parse_args()
    main(args)