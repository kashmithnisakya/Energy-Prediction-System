from pydantic import BaseModel, validator
from src.config import config

class PredictionRequest(BaseModel):
    csv_path: str
    factory_name: str

    @validator('factory_name')
    def validate_factory_name(cls, v):
        if v.lower() not in config.VALID_FACTORIES:
            raise ValueError(f"Factory name must be one of {config.VALID_FACTORIES}")
        return v.lower()

    @validator('csv_path')
    def validate_csv_path(cls, v):
        if not v.endswith('.csv'):
            raise ValueError("File path must end with .csv")
        return v