"""
FastAPI Application for Credit Default Prediction

This API serves predictions from the trained Spark ML model.
"""

import os
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel

from schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gbt_credit_default")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")

# Global variables for model and Spark session
spark: Optional[SparkSession] = None
preprocessing_model: Optional[PipelineModel] = None
gbt_model: Optional[GBTClassificationModel] = None
model_metadata: Optional[dict] = None


def initialize_spark():
    """Initialize Spark session"""
    global spark
    
    logger.info("Initializing Spark session...")
    
    spark = (
        SparkSession.builder
        .appName("credit-default-api")
        .master("local[*]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.shuffle.partitions", "10")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    
    # Set log level to WARN to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Spark session initialized: {spark.version}")
    return spark


def load_models():
    """Load indexer, scaler, and GBT model"""
    global preprocessing_model, gbt_model, model_metadata
    
    logger.info(f"Loading models from {MODEL_DIR}...")
    
    indexer_path = f"{MODEL_DIR}/{MODEL_NAME}_indexer"
    scaler_path = f"{MODEL_DIR}/{MODEL_NAME}_scaler"
    gbt_path = f"{MODEL_DIR}/{MODEL_NAME}_gbt"
    metadata_path = f"{MODEL_DIR}/{MODEL_NAME}_metadata.json"
    assembler_config_path = f"{MODEL_DIR}/{MODEL_NAME}_assembler_config.json"
    
    # Check if model files exist
    if not os.path.exists(indexer_path):
        raise FileNotFoundError(f"Indexer model not found at {indexer_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler model not found at {scaler_path}")
    if not os.path.exists(gbt_path):
        raise FileNotFoundError(f"GBT model not found at {gbt_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    if not os.path.exists(assembler_config_path):
        raise FileNotFoundError(f"Assembler config not found at {assembler_config_path}")
    
    # Load models
    from pyspark.ml.feature import StandardScalerModel, VectorAssembler
    
    indexer_model = PipelineModel.load(indexer_path)
    scaler_model = StandardScalerModel.load(scaler_path)
    gbt_model = GBTClassificationModel.load(gbt_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        model_metadata = json.load(f)
    
    # Load assembler config
    with open(assembler_config_path, 'r') as f:
        assembler_config = json.load(f)
    
    # Create assembler from config
    assembler = VectorAssembler(
        inputCols=assembler_config['inputCols'],
        outputCol=assembler_config['outputCol']
    )
    
    # Store all components in a dict for preprocessing
    preprocessing_model = {
        'indexer': indexer_model,
        'assembler': assembler,
        'scaler': scaler_model
    }
    
    logger.info("Models loaded successfully!")
    logger.info(f"Model created at: {model_metadata.get('created_at', 'unknown')}")
    logger.info(f"AUC: {model_metadata.get('metrics', {}).get('auc', 'unknown')}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting up API...")
    try:
        initialize_spark()
        load_models()
        logger.info("API startup complete!")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    if spark:
        spark.stop()
        logger.info("Spark session stopped")


# Create FastAPI app
app = FastAPI(
    title="Credit Default Prediction API",
    description="REST API for predicting credit card default risk using Apache Spark ML",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_risk_level(probability: float) -> str:
    """Determine risk level based on default probability"""
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    else:
        return "high"


def create_spark_dataframe(features_dict: dict):
    """Create Spark DataFrame from features dictionary"""
    
    # Get feature columns from metadata
    feature_metadata = model_metadata['feature_metadata']
    numeric_cols = feature_metadata['numeric_cols']
    categorical_cols = feature_metadata['categorical_cols']
    
    # Create schema
    fields = []
    for col in numeric_cols:
        fields.append(StructField(col, DoubleType(), True))
    for col in categorical_cols:
        fields.append(StructField(col, StringType(), True))
    
    schema = StructType(fields)
    
    # Prepare row data
    row_data = {}
    for col in numeric_cols + categorical_cols:
        value = features_dict.get(col)
        if value is not None:
            row_data[col] = value
        else:
            row_data[col] = None
    
    # Create DataFrame
    df = spark.createDataFrame([row_data], schema=schema)
    
    return df


def make_prediction(features_dict: dict) -> tuple[int, float]:
    """Make a single prediction"""
    
    # Create Spark DataFrame
    df = create_spark_dataframe(features_dict)
    
    # Apply preprocessing steps (matching the training pipeline)
    # 1. Apply indexer (categorical encoding)
    df_indexed = preprocessing_model['indexer'].transform(df)
    
    # 2. Apply assembler (combine features into vector)
    df_assembled = preprocessing_model['assembler'].transform(df_indexed)
    
    # 3. Apply scaler (standardize features)
    df_scaled = preprocessing_model['scaler'].transform(df_assembled)
    
    # 4. Make prediction with GBT model
    prediction_df = gbt_model.transform(df_scaled)
    
    # Extract results
    result = prediction_df.select("prediction", "probability").collect()[0]
    prediction = int(result['prediction'])
    
    # Get probability of class 1 (default)
    probability_vector = result['probability']
    probability = float(probability_vector[1])
    
    return prediction, probability


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check and API information"""
    return HealthResponse(
        status="healthy",
        model_loaded=preprocessing_model is not None and gbt_model is not None,
        spark_active=spark is not None and not spark._jsc.sc().isStopped(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=preprocessing_model is not None and gbt_model is not None,
        spark_active=spark is not None and not spark._jsc.sc().isStopped(),
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model metadata and information"""
    
    if model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not loaded"
        )
    
    feature_metadata = model_metadata['feature_metadata']
    
    return ModelInfo(
        model_name=model_metadata['model_name'],
        version="1.0.0",
        created_at=model_metadata['created_at'],
        metrics=model_metadata['metrics'],
        feature_count=len(feature_metadata['feature_cols']),
        categorical_features=feature_metadata['categorical_cols'],
        numeric_features_count=len(feature_metadata['numeric_cols'])
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction for credit default risk
    
    Returns:
    - prediction: 0 (no default) or 1 (default)
    - probability: Probability of default (0.0 to 1.0)
    - risk_level: low, medium, or high
    """
    
    try:
        # Make prediction
        prediction, probability = make_prediction(request.customer.features)
        
        # Determine risk level
        risk_level = get_risk_level(probability)
        
        return PredictionResponse(
            customer_id=request.customer.customer_id,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple customers
    
    Processes up to 1000 customers in a single request
    """
    
    try:
        predictions = []
        
        for customer in request.customers:
            # Make prediction
            prediction, probability = make_prediction(customer.features)
            
            # Determine risk level
            risk_level = get_risk_level(probability)
            
            predictions.append(
                PredictionResponse(
                    customer_id=customer.customer_id,
                    prediction=prediction,
                    probability=probability,
                    risk_level=risk_level
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
