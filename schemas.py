"""
Pydantic schemas for API request/response validation
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator


class CustomerFeatures(BaseModel):
    """Features for a single customer prediction"""
    
    # Note: In production, you would list all 190+ features here
    # For now, we'll use a flexible dict approach
    features: Dict[str, Optional[float]] = Field(
        ...,
        description="Dictionary of feature names to values. Numeric features should be float, categorical can be string or int.",
        example={
            "B_1": 0.5,
            "B_2": 1.2,
            "B_30": "A",
            "D_39": 0.8,
            # ... more features
        }
    )
    
    customer_id: Optional[str] = Field(
        None,
        description="Optional customer ID for tracking"
    )
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class PredictionRequest(BaseModel):
    """Single prediction request"""
    
    customer: CustomerFeatures = Field(
        ...,
        description="Customer features for prediction"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    
    customers: List[CustomerFeatures] = Field(
        ...,
        description="List of customers for batch prediction",
        min_items=1,
        max_items=1000  # Limit batch size
    )
    
    @validator('customers')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 customers")
        return v


class PredictionResponse(BaseModel):
    """Prediction response for a single customer"""
    
    customer_id: Optional[str] = Field(
        None,
        description="Customer ID if provided in request"
    )
    
    prediction: int = Field(
        ...,
        description="Binary prediction: 0 (no default) or 1 (default)",
        ge=0,
        le=1
    )
    
    probability: float = Field(
        ...,
        description="Probability of default (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    risk_level: str = Field(
        ...,
        description="Risk level: low, medium, or high"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each customer"
    )
    
    total_processed: int = Field(
        ...,
        description="Total number of customers processed"
    )


class ModelInfo(BaseModel):
    """Model metadata and information"""
    
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    created_at: str = Field(..., description="Model creation timestamp")
    
    metrics: Dict[str, float] = Field(
        ...,
        description="Model performance metrics",
        example={
            "auc": 0.85,
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72
        }
    )
    
    feature_count: int = Field(
        ...,
        description="Total number of features"
    )
    
    categorical_features: List[str] = Field(
        ...,
        description="List of categorical feature names"
    )
    
    numeric_features_count: int = Field(
        ...,
        description="Number of numeric features"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    spark_active: bool = Field(..., description="Whether Spark session is active")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
