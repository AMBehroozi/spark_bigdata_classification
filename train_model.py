"""
Model Training Script for Credit Default Prediction

This script extracts the training logic from the Jupyter notebook and creates
a standalone training pipeline that saves the model for API deployment.
"""

import os
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Configuration
SPARK_CORES = 12
DRIVER_MEM = "64g"
SHUFFLE_PARTS = 320
DATA_PATH = "data/amex-raddar-parquet/data_clean.parquet"
MODEL_OUTPUT_DIR = "models"
MODEL_NAME = "gbt_credit_default"

def initialize_spark():
    """Initialize Spark session with optimized configurations"""
    print("Initializing Spark session...")
    
    # Stop any existing session
    try:
        spark = SparkSession.getActiveSession()
        if spark:
            spark.stop()
            print("Stopped existing Spark session.")
    except:
        pass
    
    spark = (
        SparkSession.builder
        .appName("credit-default-training")
        .master(f"local[{SPARK_CORES}]")
        .config("spark.driver.memory", DRIVER_MEM)
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTS))
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    
    print(f"Spark session initialized: {spark.version}")
    return spark


def load_and_prepare_data(spark):
    """Load data and perform initial preparation"""
    print(f"\nLoading data from {DATA_PATH}...")
    
    df = spark.read.parquet(DATA_PATH)
    
    # Repartition and cache
    print(f"Repartitioning to {SHUFFLE_PARTS} partitions and caching...")
    df = df.repartition(SHUFFLE_PARTS).cache()
    
    row_count = df.count()
    print(f"Data loaded: {row_count:,} rows, {len(df.columns)} columns")
    
    return df


def define_feature_columns(df):
    """Define categorical and numeric feature columns"""
    
    # Define column groups
    id_cols = ['customer_ID', 'date', 'test']
    label_col = 'target'
    
    categorical_cols = [
        'B_30', 'B_38', 'D_114', 'D_116', 'D_117',
        'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'
    ]
    
    # Filter to existing categorical columns
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    # Auto-detect numeric columns
    numeric_cols = [
        c for c, t in df.dtypes
        if c not in id_cols + [label_col] + categorical_cols
        and t in ('int', 'bigint', 'float', 'double', 'smallint', 'tinyint')
    ]
    
    feature_cols = numeric_cols + categorical_cols
    
    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"  - Numeric: {len(numeric_cols)}")
    print(f"  - Categorical: {len(categorical_cols)}")
    
    return {
        'id_cols': id_cols,
        'label_col': label_col,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'feature_cols': feature_cols
    }


def clean_data(df, feature_metadata):
    """Clean data by removing rows with all features missing"""
    print("\nCleaning data...")
    
    feature_cols = feature_metadata['feature_cols']
    numeric_cols = feature_metadata['numeric_cols']
    categorical_cols = feature_metadata['categorical_cols']
    
    # Build condition to drop rows where ALL features are null/NaN
    conditions = (
        [F.col(c).isNull() for c in categorical_cols] +
        [F.col(c).isNull() | F.isnan(c) for c in numeric_cols]
    )
    
    all_null_condition = conditions[0]
    for cond in conditions[1:]:
        all_null_condition = all_null_condition & cond
    
    initial_count = df.count()
    df_clean = df.filter(~all_null_condition)
    final_count = df_clean.count()
    
    print(f"Removed {initial_count - final_count:,} rows with all features missing")
    print(f"Remaining rows: {final_count:,}")
    
    return df_clean


def build_preprocessing_pipeline(feature_metadata):
    """Build the preprocessing pipeline"""
    print("\nBuilding preprocessing pipeline...")
    
    numeric_cols = feature_metadata['numeric_cols']
    categorical_cols = feature_metadata['categorical_cols']
    
    stages = []
    
    # 1. Impute numeric features
    if numeric_cols:
        imputer_numeric = Imputer(
            inputCols=numeric_cols,
            outputCols=[f"{c}_imputed" for c in numeric_cols],
            strategy="mean"
        )
        stages.append(imputer_numeric)
    
    # 2. Index categorical features
    indexed_cat_cols = []
    for col in categorical_cols:
        indexer = StringIndexer(
            inputCol=col,
            outputCol=f"{col}_indexed",
            handleInvalid="keep"
        )
        stages.append(indexer)
        indexed_cat_cols.append(f"{col}_indexed")
    
    # 3. Assemble all features
    imputed_numeric_cols = [f"{c}_imputed" for c in numeric_cols]
    all_feature_cols = imputed_numeric_cols + indexed_cat_cols
    
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    stages.append(assembler)
    
    # 4. Scale features
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    stages.append(scaler)
    
    pipeline = Pipeline(stages=stages)
    
    print(f"Pipeline stages: {len(stages)}")
    return pipeline


def train_model(df_train, df_test):
    """Train the GBT classifier"""
    print("\nTraining GBT model...")
    
    # Calculate class weights
    label_counts = df_train.groupBy("target").count().collect()
    total = sum([row['count'] for row in label_counts])
    
    class_weights = {}
    for row in label_counts:
        label = row['target']
        count = row['count']
        weight = total / (2.0 * count)
        class_weights[label] = weight
    
    print(f"Class weights: {class_weights}")
    
    # Add weight column
    weight_map = F.create_map([F.lit(x) for pair in class_weights.items() for x in pair])
    df_train_weighted = df_train.withColumn("weight", weight_map[F.col("target")])
    
    # Define GBT model
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="target",
        weightCol="weight",
        maxIter=100,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )
    
    # Train model
    start_time = datetime.now()
    model = gbt.fit(df_train_weighted)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model


def evaluate_model(model, df_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    predictions = model.transform(df_test)
    
    # Calculate AUC
    evaluator = BinaryClassificationEvaluator(
        labelCol="target",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)
    
    # Calculate confusion matrix metrics
    tp = predictions.filter((F.col("target") == 1) & (F.col("prediction") == 1)).count()
    tn = predictions.filter((F.col("target") == 0) & (F.col("prediction") == 0)).count()
    fp = predictions.filter((F.col("target") == 0) & (F.col("prediction") == 1)).count()
    fn = predictions.filter((F.col("target") == 1) & (F.col("prediction") == 0)).count()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    }
    
    print(f"\nModel Performance:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return metrics


def save_artifacts(preprocessing_pipeline, model, feature_metadata, metrics):
    """Save all model artifacts"""
    print(f"\nSaving model artifacts to {MODEL_OUTPUT_DIR}/...")
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    model_path = f"{MODEL_OUTPUT_DIR}/{MODEL_NAME}"
    
    # Save preprocessing pipeline
    pipeline_path = f"{model_path}_preprocessing"
    preprocessing_pipeline.save(pipeline_path)
    print(f"  Saved preprocessing pipeline to {pipeline_path}")
    
    # Save GBT model
    gbt_model_path = f"{model_path}_gbt"
    model.save(gbt_model_path)
    print(f"  Saved GBT model to {gbt_model_path}")
    
    # Save metadata
    metadata = {
        'model_name': MODEL_NAME,
        'created_at': datetime.now().isoformat(),
        'feature_metadata': feature_metadata,
        'metrics': metrics,
        'spark_config': {
            'cores': SPARK_CORES,
            'driver_memory': DRIVER_MEM,
            'shuffle_partitions': SHUFFLE_PARTS
        }
    }
    
    metadata_path = f"{model_path}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")
    
    print("\n✅ All artifacts saved successfully!")
    return model_path


def main():
    """Main training pipeline"""
    print("="*80)
    print("Credit Default Prediction - Model Training")
    print("="*80)
    
    # Initialize Spark
    spark = initialize_spark()
    
    try:
        # Load data
        df = load_and_prepare_data(spark)
        
        # Define features
        feature_metadata = define_feature_columns(df)
        
        # Clean data
        df_clean = clean_data(df, feature_metadata)
        
        # Split data
        print("\nSplitting data (80/20 train/test)...")
        train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)
        train_df.cache()
        test_df.cache()
        
        print(f"Training set: {train_df.count():,} rows")
        print(f"Test set: {test_df.count():,} rows")
        
        # Build and fit preprocessing pipeline
        preprocessing_pipeline = build_preprocessing_pipeline(feature_metadata)
        print("\nFitting preprocessing pipeline...")
        preprocessing_model = preprocessing_pipeline.fit(train_df)
        
        # Transform data
        print("Transforming training data...")
        train_processed = preprocessing_model.transform(train_df)
        print("Transforming test data...")
        test_processed = preprocessing_model.transform(test_df)
        
        # Train model
        model = train_model(train_processed, test_processed)
        
        # Evaluate model
        metrics = evaluate_model(model, test_processed)
        
        # Save artifacts
        model_path = save_artifacts(preprocessing_model, model, feature_metadata, metrics)
        
        print("\n" + "="*80)
        print("Training Complete!")
        print(f"Model saved to: {model_path}")
        print("="*80)
        
    finally:
        # Clean up
        spark.stop()
        print("\nSpark session stopped.")


if __name__ == "__main__":
    main()
