#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta
from prophet.serialize import model_to_json

# ----------------------------------------
# Setup logging
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------------------------------
# Data loading & preprocessing
# ----------------------------------------
def load_and_preprocess_data(csv_path, test_size_days=30):
    df = pd.read_csv(csv_path, parse_dates=['time'])
    df = df.rename(columns={'time': 'ds', 'energy': 'y'})
    df = df.sort_values('ds')

    # Replace zeros and forward-fill
    df['y'] = df['y'].replace(0, np.nan).ffill()
    # Split into train/test by date
    split_date = df['ds'].max() - timedelta(days=test_size_days)
    train = df[df['ds'] <= split_date].copy()
    test  = df[df['ds'] >  split_date].copy()
    logger.info(f"Training on {len(train)} points; testing on {len(test)} points")
    return train, test, df

# ----------------------------------------
# Model builder
# ----------------------------------------
def build_prophet_model(params):
    m = Prophet(
        growth=params.get('growth', 'linear'),
        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
        holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
        seasonality_mode=params.get('seasonality_mode', 'additive'),
        yearly_seasonality=params.get('yearly_seasonality', 'auto'),
        weekly_seasonality=params.get('weekly_seasonality', 'auto'),
        daily_seasonality=params.get('daily_seasonality', False),
        interval_width=params.get('interval_width', 0.95)
    )
    # Add custom seasonalities
    for s in params.get('custom_seasonalities', []):
        m.add_seasonality(name=s['name'], period=s['period'], fourier_order=s['fourier_order'])
    # Add country holidays
    if params.get('country_holidays'):
        m.add_country_holidays(country_name=params['country_holidays'])
    return m

# ----------------------------------------
# Hyperparameter tuning via cross-validation
# ----------------------------------------
def tune_hyperparameters(train, base_params, grid, cv_params):
    logger.info("Starting hyperparameter tuning...")
    best_score = float('inf')
    best_params = {}

    for cps in grid['changepoint_prior_scale']:
        for sps in grid['seasonality_prior_scale']:
            params = base_params.copy()
            params['changepoint_prior_scale']   = cps
            params['seasonality_prior_scale'] = sps

            model = build_prophet_model(params)
            model.fit(train)
            df_cv = cross_validation(
                model,
                initial=cv_params['initial'],
                period=cv_params['period'],
                horizon=cv_params['horizon']
            )
            df_p = performance_metrics(df_cv)
            mape = df_p['mape'].mean()
            logger.info(f"cps={cps}, sps={sps} -> Mean MAPE={mape:.3f}")
            if mape < best_score:
                best_score = mape
                best_params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps}

    logger.info(f"Best hyperparameters: {best_params}, MAPE={best_score:.3f}")
    return best_params

# ----------------------------------------
# Evaluation
# ----------------------------------------
def evaluate_model(model, test):
    future = model.make_future_dataframe(periods=len(test), freq='D', include_history=False)
    fc = model.predict(future)

    # Align dates
    fc['ds'] = fc['ds'].dt.date
    test['ds'] = test['ds'].dt.date
    merged = fc[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds')

    mae  = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100

    logger.info(f"Evaluation -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    return fc, merged

# ----------------------------------------
# Plotting
# ----------------------------------------
def plot_forecast(model, forecast, test, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    fig = model.plot(forecast)
    ax  = fig.gca()
    dates = test['ds'].apply(lambda x: x if isinstance(x, pd.Timestamp) else pd.to_datetime(x))
    ax.plot(dates, test['y'], 'r.', label='Actual')
    plt.title('Energy Consumption Forecast')
    plt.savefig(os.path.join(save_dir, 'forecast.png'))
    plt.close()

    comp = model.plot_components(forecast)
    comp.savefig(os.path.join(save_dir, 'components.png'))
    plt.close()

    # Prediction vs Actual
    plt.figure(figsize=(10,5))
    plt.plot(dates, test['y'], label='Actual')
    plt.plot(dates, forecast['yhat'], label='Predicted')
    plt.legend(); plt.grid(True)
    plt.title('Prediction vs Actual')
    plt.savefig(os.path.join(save_dir, 'prediction_vs_actual.png'))
    plt.close()

# ----------------------------------------
# Model saving
# ----------------------------------------
def save_model(model, name, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    path = Path(save_dir) / f"{name}.json"
    with open(path, 'w') as fp:
        fp.write(model_to_json(model))
    logger.info(f"Model saved to {path}")

# ----------------------------------------
# Main entrypoint
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prophet energy forecasting")
    parser.add_argument('--csv_path', required=True, help='Path to input CSV')
    parser.add_argument('--test_days', type=int, default=30, help='Days for test split')
    parser.add_argument('--tune',    action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--model_name', default='energy_prophet_v1_test', help='Name for saved model')
    args = parser.parse_args()

    train, test, _ = load_and_preprocess_data(args.csv_path, args.test_days)

    # Base Prophet parameters
    base_params = {
        'growth': 'linear',
        'holidays_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': False,
        'custom_seasonalities': [
            {'name': 'monthly', 'period': 30.5, 'fourier_order': 5},
            {'name': 'quarterly', 'period': 91.25, 'fourier_order': 8}
        ],
        'country_holidays': 'LK',
        'interval_width': 0.95,
        # initial defaults (will be overridden by tuning)
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0
    }

    # Optional hyperparameter grid
    grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1],
        'seasonality_prior_scale': [5.0, 10.0, 15.0]
    }
    cv_params = {'initial': '250 days', 'period': '60 days', 'horizon': '30 days'}

    if args.tune:
        best = tune_hyperparameters(train, base_params, grid, cv_params)
        base_params.update(best)

    # Fit final model
    model = build_prophet_model(base_params)
    model.fit(train)

    # Evaluate
    forecast, merged = evaluate_model(model, test)

    # Plot
    plot_forecast(model, forecast, test)

    # Save
    save_model(model, args.model_name)
    pd.DataFrame(forecast)[['ds','yhat','yhat_lower','yhat_upper']].to_csv(f"{args.model_name}_forecast.csv", index=False)
    logger.info("Forecast CSV saved.")

if __name__ == "__main__":
    main()
