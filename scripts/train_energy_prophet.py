import argparse
import pandas as pd
import numpy as np
import mlflow
import logging
import os
from pathlib import Path
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowProphetLogger:
    def __init__(self, model_name, save_dir="models"):
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def log_model(self, model, metrics=None):
        """Log Prophet model and metrics to MLflow"""
        if metrics:
            mlflow.log_metrics(metrics)
        
        # Save model
        model_path = os.path.join(self.save_dir, f"{self.model_name}.json")
        model_serialized = model_to_json(model)
        with open(model_path, 'w') as f:
            f.write(model_serialized)
        mlflow.log_artifact(model_path, artifact_path="models")

def model_to_json(model):
    """Serialize Prophet model to JSON"""
    from prophet.serialize import model_to_json
    return model_to_json(model)

def json_to_model(model_str):
    """Deserialize Prophet model from JSON"""
    from prophet.serialize import model_from_json
    return model_from_json(model_str)

def load_and_preprocess_data(csv_path, test_size_days=30):
    """Load and preprocess data for Prophet"""
    try:
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['time'])
        df = df.rename(columns={'time': 'ds', 'energy': 'y'})
        df = df.sort_values('ds')
        
        # Handle zeros and missing values
        first_valid = df['y'].ne(0).idxmax()
        df.loc[:first_valid, 'y'] = np.nan
        df['y'] = df['y'].replace(0, np.nan).ffill()
        
        # Train-test split
        split_date = df['ds'].max() - timedelta(days=test_size_days)
        train = df[df['ds'] <= split_date]
        test = df[df['ds'] > split_date]
        
        logger.info(f"Train period: {train['ds'].min()} to {train['ds'].max()}")
        logger.info(f"Test period: {test['ds'].min()} to {test['ds'].max()}")
        
        return train, test, df
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def build_prophet_model(params):
    """Initialize Prophet model with given parameters"""
    logger.info("Building Prophet model")
    model = Prophet(
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
    
    # Add custom seasonalities if specified
    if 'custom_seasonalities' in params:
        for seasonality in params['custom_seasonalities']:
            model.add_seasonality(
                name=seasonality['name'],
                period=seasonality['period'],
                fourier_order=seasonality['fourier_order']
            )
    
    # Add holidays if specified
    if 'country_holidays' in params and params['country_holidays']:
        model.add_country_holidays(country_name=params['country_holidays'])
    
    return model

def evaluate_model(model, test):
    """Evaluate model performance on test set"""
    try:
        # Create future dataframe for test period
        future = model.make_future_dataframe(periods=len(test), freq='D', include_history=False)
        
        # Predict
        forecast = model.predict(future)
        
        # Align datetime formats to dates only so merge works as expected
        forecast['ds'] = forecast['ds'].dt.date
        test['ds'] = test['ds'].dt.date
        
        # Merge forecast with actual test data
        results = forecast[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds')
        
        # Calculate metrics
        mae = mean_absolute_error(results['y'], results['yhat'])
        rmse = np.sqrt(mean_squared_error(results['y'], results['yhat']))
        mape = np.mean(np.abs((results['y'] - results['yhat']) / results['y'])) * 100
        
        metrics = {
            'test_mae': mae,
            'test_rmse': rmse,
            'test_mape': mape
        }
        
        logger.info("\nModel Evaluation:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
        
        return metrics, results
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def plot_results(model, forecast, test, save_dir="plots"):
    """Generate and save visualization plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Forecast plot
        fig1 = model.plot(forecast)
        plt.title('Energy Consumption Forecast')
        plt.xlabel('Date')
        plt.ylabel('Energy')
        # Convert test ds to datetime.date if not already
        test_dates = test['ds'].apply(lambda x: x.date() if isinstance(x, pd.Timestamp) else x)
        ax = fig1.gca()
        ax.plot(test_dates, test['y'], 'r.', label='Actual')
        plt.legend()
        forecast_path = os.path.join(save_dir, "forecast_plot.png")
        plt.savefig(forecast_path)
        plt.close()
        mlflow.log_artifact(forecast_path)
        
        # Components plot
        fig2 = model.plot_components(forecast)
        components_path = os.path.join(save_dir, "components_plot.png")
        plt.savefig(components_path)
        plt.close()
        mlflow.log_artifact(components_path)
        
        # Prediction vs Actual plot
        # Align forecast dates to dates only for merging
        forecast['ds'] = forecast['ds'].dt.date
        test['ds'] = test['ds'].dt.date
        merged = forecast[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds')
        plt.figure(figsize=(12, 6))
        plt.plot(merged['ds'], merged['y'], label='Actual')
        plt.plot(merged['ds'], merged['yhat'], label='Predicted')
        plt.title('Energy Prediction vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        pred_path = os.path.join(save_dir, "prediction_plot.png")
        plt.savefig(pred_path)
        plt.close()
        mlflow.log_artifact(pred_path)
        
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")

def cross_validate_model(model, train, params):
    """Perform time series cross-validation"""
    try:
        initial = params.get('cv_initial', '250 days')
        period = params.get('cv_period', '60 days')
        horizon = params.get('cv_horizon', '30 days')
        
        df_cv = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        df_p = performance_metrics(df_cv)
        
        # Log CV metrics
        cv_metrics = {
            'cv_mae': df_p['mae'].mean(),
            'cv_rmse': df_p['rmse'].mean(),
            'cv_mape': df_p['mape'].mean()
        }
        
        mlflow.log_metrics(cv_metrics)
        
        logger.info("\nCross-Validation Metrics:")
        logger.info(df_p.describe())
        
        return df_p
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        return None

def main(args):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(vars(args))
        
        # Data loading
        train, test, full_df = load_and_preprocess_data(
            args.csv_path,
            test_size_days=args.test_size_days
        )
        
        # Model configuration
        prophet_params = {
            'growth': args.growth,
            'changepoint_prior_scale': args.changepoint_prior_scale,
            'seasonality_prior_scale': args.seasonality_prior_scale,
            'yearly_seasonality': args.yearly_seasonality,
            'weekly_seasonality': args.weekly_seasonality,
            'daily_seasonality': args.daily_seasonality,
            'custom_seasonalities': [
                {'name': 'monthly', 'period': 30.5, 'fourier_order': 5},
                {'name': 'quarterly', 'period': 91.25, 'fourier_order': 8}
            ],
            'country_holidays': args.country_holidays,
            'cv_initial': args.cv_initial,
            'cv_period': args.cv_period,
            'cv_horizon': args.cv_horizon
        }
        
        # Build and train model
        model = build_prophet_model(prophet_params)
        model.fit(train)
        
        # Cross-validation
        cv_results = cross_validate_model(model, train, prophet_params)
        
        # Evaluation
        test_metrics, test_results = evaluate_model(model, test)
        
        # Generate full forecast for visualization
        future = model.make_future_dataframe(periods=args.test_size_days)
        forecast = model.predict(future)
        
        # Logging and saving
        mlflow_logger = MLflowProphetLogger(args.model_name)
        mlflow_logger.log_model(model, test_metrics)
        
        # Visualization
        plot_results(model, forecast, test)
        
        # Save full results
        results_path = Path(f"{args.model_name}_results.csv")
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Prophet model for energy prediction")
    
    # Data parameters
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--test_size_days', type=int, default=30, help='Days to use for testing')
    
    # Model parameters
    parser.add_argument('--growth', type=str, default='linear', choices=['linear', 'flat', 'logistic'], help='Growth type')
    parser.add_argument('--changepoint_prior_scale', type=float, default=0.05, help='Changepoint flexibility')
    parser.add_argument('--seasonality_prior_scale', type=float, default=10.0, help='Seasonality flexibility')
    parser.add_argument('--yearly_seasonality', type=str, default='auto', help='Yearly seasonality')
    parser.add_argument('--weekly_seasonality', type=str, default='auto', help='Weekly seasonality')
    parser.add_argument('--daily_seasonality', type=bool, default=False, help='Daily seasonality')
    # Updated default to "LK" for Sri Lanka holidays
    parser.add_argument('--country_holidays', type=str, default='LK', help='Country code for holidays (e.g., LK for Sri Lanka)')
    
    # Cross-validation parameters
    parser.add_argument('--cv_initial', type=str, default='250 days', help='Initial period for cross validation')
    parser.add_argument('--cv_period', type=str, default='60 days', help='Period between cross validation cutoffs')
    parser.add_argument('--cv_horizon', type=str, default='30 days', help='Horizon for cross validation')
    
    # Output parameters
    parser.add_argument('--model_name', type=str, default="prophet_energy", help='Name for saved model')
    
    args = parser.parse_args()
    main(args)



# CLI to run the script:
# python train_energy_prophet.py \
#     --csv_path ../ml/data/processed/china_mill_data_2025_03_04_09_30_30.csv \
#     --test_size_days 60 \
#     --changepoint_prior_scale 0.1 \
#     --seasonality_prior_scale 15.0 \
#     --country_holidays LK \
#     --model_name energy_prophet_v1_test