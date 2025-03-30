# LSTM Prediction API

## Overview
This is a FastAPI-based API for predicting power and energy consumption using an LSTM model. It leverages TensorFlow with GPU support for accelerated predictions and is containerized using Docker for easy deployment.

## Features
- Predicts energy and power consumption from time-series CSV data
- Uses a pre-trained LSTM model
- GPU acceleration with TensorFlow
- RESTful API endpoints
- Health check monitoring
- Temporary file storage for prediction outputs

## Project Structure
```bash
project/
├── src/                 # Application source code
│   ├── main.py          # FastAPI app setup
│   ├── config.py        # Configuration settings
│   ├── models.py        # Pydantic models
│   ├── services/        # Business logic
│   └── utils/           # Utility functions
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```
## Running the Application

### Using Docker Compose (Recommended)
```bash
# Build and run the container
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Using Docker Directly
```bash
# Build the image
docker build -t lstm-prediction-api .

# Run the container
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/temp:/app/temp \
  -v $(pwd)/../ml/model:/app/ml/model \
  --name lstm_prediction_api \
  lstm-prediction-api
```

The API will be available at http://localhost:8000.

## API Endpoints

### POST /predict/
* **Description**: Make predictions using a CSV file path and factory name
* **Request Body**:
```json
{
  "csv_path": "/app/temp/input.csv",
  "factory_name": "factory1"
}
```
* **Response**: Downloadable CSV file with predictions
* **Notes**:
   * csv_path must be accessible within the container (e.g., via mounted volume)
   * CSV must contain time, energy, and power columns
   * Minimum 672 rows required

### GET /health
* **Description**: Check API health status
* **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Example Usage
1. Place an input CSV file in the temp/ directory
2. Make a prediction request:
```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "/app/temp/input.csv", "factory_name": "factory1"}' \
  --output predictions.csv
```
3. Check the downloaded predictions.csv file

## Configuration
* **Model Path**: /app/ml/model/lstm_model.h5 (mounted from host)
* **Temp Directory**: /app/temp/ (mounted from ./temp)
* **Port**: 8000 (configurable via docker-compose.yml)
* **Log Level**: Set via LOG_LEVEL environment variable (default: INFO)

## Requirements
* Python 3.9 (provided by Docker image)
* TensorFlow 2.12.0 with GPU support (included in base image)
* CSV input files must have:
   * time column (parseable as datetime)
   * energy and power columns
   * At least 672 rows of data
* Supported factory names: factory1, factory2, factory3, factory4, factory5

## Troubleshooting
* **GPU Not Detected**: Verify NVIDIA drivers and Container Toolkit installation
* **Model Not Found**: Check volume mapping and model file location
* **Permission Issues**: Ensure Docker has access to mounted directories
* **API Not Responding**: Check container logs with docker-compose logs

## Development
To modify the application:
1. Edit files in src/
2. Rebuild and restart: docker-compose up -d --build