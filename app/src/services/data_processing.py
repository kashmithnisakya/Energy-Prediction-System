import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from fastapi import HTTPException
from src.config import config
from src.utils.logging import logger

def load_and_preprocess_data(file_path: Path, features: list[str], window_size: int, scaler=None):
    """Load and preprocess data for prediction"""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['time'])
        df = df.sort_values('time').set_index('time')
        df = df[features]
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

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

    if scaler is None:
        logger.info("Creating new scaler")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
    else:
        scaled_data = scaler.transform(df)

    if len(scaled_data) < window_size:
        raise HTTPException(status_code=400, detail=f"Data too short. Minimum {window_size} rows required.")
    
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
    
    X = np.array(X)
    logger.info(f"Created prediction sequences with shape: {X.shape}")
    return X, df.index[window_size:], scaler