"""
Model Training Script for Credit Default Prediction

This script extracts the training logic from the Jupyter notebook and creates
a standalone training pipeline that saves the model for API deployment.

IMPORTANT: This script matches the exact data cleaning and preprocessing logic
from American_Express.ipynb to ensure consistency.
"""

import os
import json
import time
from datetime import datetime
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
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


def clean_data(df):
    """
    Clean data using the EXACT same logic as American_Express.ipynb
    1) Drop rows where ALL features are null/NaN
    2) Drop columns with >95% missing values
    3) Repartition and cache
    """
    print("\nStarting data cleaning...")
    
    # --- Define column groups (EXACTLY as in notebook) ---
    id_cols = ['customer_ID', 'date', 'test']
    label_col = 'target'
    categorical_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117',
                        'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    
    # Filter to existing categorical columns
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    # Auto-detect numeric columns
    numeric_cols = [
        c for c, t in df.dtypes
        if c not in id_cols + [label_col] + categorical_cols
        and t in ('int', 'bigint', 'float', 'double', 'smallint', 'tinyint')
    ]
    
    feature_cols = numeric_cols + categorical_cols
    print(f"Total feature columns: {len(feature_cols)} (numeric: {len(numeric_cols)}, cat: {len(categorical_cols)})")
    
    # --------------------------------------------------------------
    # 1) Drop rows where ALL feature columns are null/NaN
    # --------------------------------------------------------------
    
    print("Building condition to drop rows with ALL features missing...")
    
    conditions = (
        [F.col(c).isNull() | F.isnan(c) for c in numeric_cols] +
        [F.col(c).isNull() for c in categorical_cols]
    )
    
    if conditions:
        all_null_cond = reduce(lambda x, y: x & y, conditions)
        df_no_allnull = df.filter(~all_null_cond)
    else:
        df_no_allnull = df  # no features → keep all
    
    rows_before = df.count()
    rows_after = df_no_allnull.count()
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after dropping all-null-feature rows: {rows_after:,}")
    
    # --------------------------------------------------------------
    # 2) Drop columns with >95% missing
    # --------------------------------------------------------------
    
    missing_threshold = 0.95
    row_count = rows_after
    
    print(f"Computing missing rates for {len(feature_cols)} feature columns...")
    
    null_exprs = []
    for c in feature_cols:
        if c in numeric_cols:
            null_exprs.append(
                F.sum((F.col(c).isNull() | F.isnan(c)).cast("int")).alias(c)
            )
        else:
            null_exprs.append(F.sum(F.col(c).isNull().cast("int")).alias(c))
    
    missing_counts = df_no_allnull.select(null_exprs).first().asDict()
    drop_cols = [
        c for c in feature_cols
        if missing_counts.get(c, 0) / row_count > missing_threshold
    ]
    
    print(f"Columns to drop (>{missing_threshold*100:.0f}% missing): {len(drop_cols)}")
    if drop_cols:
        print(drop_cols)
    
    df_clean = df_no_allnull.drop(*drop_cols)
    
    print(f"Final number of columns: {len(df_clean.columns)}")
    
    # --------------------------------------------------------------
    # 3) REPARTITION + CACHE df_clean
    # --------------------------------------------------------------
    
    print("Optimizing df_clean for maximum performance...")
    df_clean = df_clean.repartition(SHUFFLE_PARTS)
    df_clean = df_clean.cache()
    
    print("Triggering cache on df_clean ...")
    start_cache = time.time()
    cached_rows = df_clean.count()
    print(f"Cache ready in {time.time() - start_cache:.1f}s")
    
    print("\n" + "="*80)
    print(f"Rows       : {cached_rows:,}")
    print(f"Columns    : {len(df_clean.columns)}")
    print(f"Partitions : {df_clean.rdd.getNumPartitions()}")
    print("="*80)
    
    # Return metadata for later use
    metadata = {
        'id_cols': id_cols,
        'label_col': label_col,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'feature_cols': feature_cols,
        'dropped_cols': drop_cols
    }
    
    return df_clean, metadata


def impute_data(df, metadata, spark):
    """
    Impute missing values using EXACT same logic as notebook
    - Numeric: mean imputation using coalesce
    - Categorical: fillna with "missing"
    """
    print("\nStarting imputation...")
    
    id_cols = metadata['id_cols']
    label_col = metadata['label_col']
    categorical_cols = metadata['categorical_cols']
    
    start = time.time()
    
    # REBUILD numeric_cols FROM THE CURRENT df_clean (CRITICAL!)
    numeric_cols = [
        c for c, t in df.dtypes
        if c not in id_cols + [label_col] + categorical_cols
        and t in ('int', 'bigint', 'float', 'double', 'smallint', 'tinyint')
    ]
    
    print(f"Using {len(numeric_cols)} numeric columns that actually exist")
    
    # ONE JOB — compute means
    print("Computing means ...")
    mean_exprs = [F.mean(c).alias(f"{c}_mean") for c in numeric_cols]
    means_row = df.select(*mean_exprs).first()
    
    mean_dict = {c: means_row[f"{c}_mean"] for c in numeric_cols}
    broadcast_means = spark.sparkContext.broadcast(mean_dict)
    
    print(f"Means computed in {time.time()-start:.1f}s")
    
    # Imputation
    print("Applying mean imputation...")
    df_imputed = (
        df.select(
            *[F.coalesce(F.col(c), F.lit(broadcast_means.value[c])).alias(c)
              if c in numeric_cols else F.col(c)
              for c in df.columns]
        )
        .fillna("missing", subset=categorical_cols)
        .cache()
    )
    
    df_imputed.count()
    print(f"\nIMPUTATION DONE IN {time.time()-start:.1f} SECONDS")
    
    # Update metadata with current numeric_cols
    metadata['numeric_cols'] = numeric_cols
    
    return df_imputed, metadata


def index_categorical(df, metadata):
    """
    Index categorical columns using StringIndexer (EXACT same as notebook)
    """
    print("\nIndexing categorical columns...")
    
    categorical_cols = metadata['categorical_cols']
    
    # New names for indexed columns
    categorical_indexed_cols = [c + "_idx" for c in categorical_cols]
    
    # One StringIndexer per categorical column
    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=c + "_idx",
            handleInvalid="keep"  # important: keep unseen/Null as a valid index
        )
        for c in categorical_cols
    ]
    
    # Build and fit pipeline
    indexer_pipeline = Pipeline(stages=indexers)
    indexer_model = indexer_pipeline.fit(df)
    df_indexed = indexer_model.transform(df)
    
    print("Indexed categorical columns added:")
    print(categorical_indexed_cols)
    
    metadata['categorical_indexed_cols'] = categorical_indexed_cols
    metadata['indexer_model'] = indexer_model
    
    return df_indexed, metadata


def assemble_and_scale(df, metadata):
    """
    VectorAssembler + StandardScaler (EXACT same as notebook)
    """
    print("\nAssembling and scaling features...")
    
    id_cols = metadata['id_cols']
    label_col = metadata['label_col']
    categorical_cols = metadata['categorical_cols']
    categorical_indexed_cols = metadata['categorical_indexed_cols']
    
    # 1) Identify numeric feature columns
    numeric_types = ('int', 'bigint', 'float', 'double', 'smallint', 'tinyint')
    
    numeric_feature_cols = [
        c for c, t in df.dtypes
        if c not in id_cols + [label_col] + categorical_cols + categorical_indexed_cols
        and t in numeric_types
    ]
    
    print("Numeric feature columns:", len(numeric_feature_cols))
    print("Categorical indexed feature columns:", len(categorical_indexed_cols))
    
    # 2) All input feature columns for the model
    feature_input_cols = numeric_feature_cols + categorical_indexed_cols
    print("Total feature columns going into assembler:", len(feature_input_cols))
    
    # 3) VectorAssembler → features_unnorm
    assembler = VectorAssembler(
        inputCols=feature_input_cols,
        outputCol="features_unnorm"
    )
    
    df_assembled = assembler.transform(df)
    
    # 4) StandardScaler → features (final feature vector)
    scaler = StandardScaler(
        inputCol="features_unnorm",
        outputCol="features",
        withStd=True,
        withMean=True  # center to mean 0
    )
    
    scaler_model = scaler.fit(df_assembled)
    df_final = scaler_model.transform(df_assembled)
    
    print("Final dataframe for modeling:")
    print(f"  Features column: 'features' (vector of {len(feature_input_cols)} elements)")
    
    metadata['assembler'] = assembler
    metadata['scaler_model'] = scaler_model
    metadata['feature_input_cols'] = feature_input_cols
    metadata['numeric_feature_cols'] = numeric_feature_cols
    
    return df_final, metadata


def train_model(df_train, df_test):
    """Train the GBT classifier with class weights"""
    print("\nTraining GBT model...")
    
    # Filter out rows with null target values
    df_train_clean = df_train.filter(F.col("target").isNotNull())
    df_test_clean = df_test.filter(F.col("target").isNotNull())
    
    print(f"Training samples after removing null targets: {df_train_clean.count():,}")
    
    # Calculate class weights
    label_counts = df_train_clean.groupBy("target").count().collect()
    total = sum([row['count'] for row in label_counts])
    
    class_weights = {}
    for row in label_counts:
        label = int(row['target'])
        count = row['count']
        weight = total / (2.0 * count)
        class_weights[label] = weight
    
    print(f"Class weights: {class_weights}")
    
    # Add weight column - use when() to avoid null keys
    weight_expr = F.when(F.col("target") == 0, F.lit(class_weights.get(0, 1.0)))
    for label, weight in class_weights.items():
        if label != 0:
            weight_expr = weight_expr.when(F.col("target") == label, F.lit(weight))
    weight_expr = weight_expr.otherwise(F.lit(1.0))
    
    df_train_weighted = df_train_clean.withColumn("weight", weight_expr)
    
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
    
    return model, df_test_clean


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


def save_artifacts(metadata, model, metrics):
    """Save all model artifacts"""
    print(f"\nSaving model artifacts to {MODEL_OUTPUT_DIR}/...")
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    model_path = f"{MODEL_OUTPUT_DIR}/{MODEL_NAME}"
    
    # Save indexer pipeline (categorical encoding)
    indexer_path = f"{model_path}_indexer"
    metadata['indexer_model'].save(indexer_path)
    print(f"  Saved indexer pipeline to {indexer_path}")
    
    # Save assembler (we'll save this as part of a pipeline)
    # Save scaler model
    scaler_path = f"{model_path}_scaler"
    metadata['scaler_model'].save(scaler_path)
    print(f"  Saved scaler model to {scaler_path}")
    
    # Save GBT model
    gbt_model_path = f"{model_path}_gbt"
    model.save(gbt_model_path)
    print(f"  Saved GBT model to {gbt_model_path}")
    
    # Save metadata (without model objects)
    metadata_to_save = {
        'model_name': MODEL_NAME,
        'created_at': datetime.now().isoformat(),
        'feature_metadata': {
            'id_cols': metadata['id_cols'],
            'label_col': metadata['label_col'],
            'categorical_cols': metadata['categorical_cols'],
            'categorical_indexed_cols': metadata['categorical_indexed_cols'],
            'numeric_cols': metadata['numeric_cols'],
            'numeric_feature_cols': metadata['numeric_feature_cols'],
            'feature_input_cols': metadata['feature_input_cols'],
            'dropped_cols': metadata.get('dropped_cols', [])
        },
        'metrics': metrics,
        'spark_config': {
            'cores': SPARK_CORES,
            'driver_memory': DRIVER_MEM,
            'shuffle_partitions': SHUFFLE_PARTS
        }
    }
    
    metadata_path = f"{model_path}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_to_save, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")
    
    # Save assembler configuration
    assembler_config = {
        'inputCols': metadata['feature_input_cols'],
        'outputCol': 'features_unnorm'
    }
    assembler_config_path = f"{model_path}_assembler_config.json"
    with open(assembler_config_path, 'w') as f:
        json.dump(assembler_config, f, indent=2)
    print(f"  Saved assembler config to {assembler_config_path}")
    
    print("\n✅ All artifacts saved successfully!")
    return model_path


def main():
    """Main training pipeline - EXACTLY matches American_Express.ipynb"""
    print("="*80)
    print("Credit Default Prediction - Model Training")
    print("(Matches American_Express.ipynb exactly)")
    print("="*80)
    
    # Initialize Spark
    spark = initialize_spark()
    
    try:
        # Load data
        df = load_and_prepare_data(spark)
        
        # Clean data (EXACT same as notebook)
        df_clean, metadata = clean_data(df)
        
        # Impute data (EXACT same as notebook)
        df_imputed, metadata = impute_data(df_clean, metadata, spark)
        
        # Index categorical columns (EXACT same as notebook)
        df_indexed, metadata = index_categorical(df_imputed, metadata)
        
        # Assemble and scale features (EXACT same as notebook)
        df_final, metadata = assemble_and_scale(df_indexed, metadata)
        
        # Split data
        print("\nSplitting data (80/20 train/test)...")
        train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)
        train_df.cache()
        test_df.cache()
        
        print(f"Training set: {train_df.count():,} rows")
        print(f"Test set: {test_df.count():,} rows")
        
        # Train model
        model, test_df_clean = train_model(train_df, test_df)
        
        # Evaluate model
        metrics = evaluate_model(model, test_df_clean)
        
        # Save artifacts
        model_path = save_artifacts(metadata, model, metrics)
        
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
