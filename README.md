# 💳 Credit Default Prediction with Apache Spark

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A **production-ready big-data machine learning pipeline** for predicting credit card default risk using **Apache Spark**. This project leverages distributed computing to process large-scale financial datasets stored in **Parquet format**, making it suitable for enterprise-level credit risk assessment.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Deployment](#-api-deployment)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Results](#-results)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project tackles the critical problem of **credit default prediction** in the financial services industry. Using anonymized customer financial profiles, the model predicts whether a customer will **default within 120 days** after their latest credit card statement.

### Key Highlights

- ⚡ **Scalable**: Built on Apache Spark for distributed processing of millions of records
- 🎯 **Accurate**: Uses Gradient Boosted Trees (GBT) for high-performance classification
- 🔧 **Production-Ready**: Complete pipeline from data ingestion to model evaluation
- 📊 **Handles Imbalance**: Implements class weighting to address imbalanced datasets
- 🚀 **Optimized**: Leverages Parquet format and Spark's adaptive query execution

---

## 📊 Dataset

### American Express Credit Default Dataset

The dataset contains **monthly customer profiles** with anonymized financial features:

- **Features**: 190+ anonymized features including:
  - Delinquency indicators
  - Account balance metrics
  - Payment history
  - Spending patterns
  - Risk scores
  
- **Target**: Binary classification (default within 120 days: Yes/No)
- **Format**: Parquet (optimized columnar storage)
- **Size**: Large-scale dataset requiring distributed processing

### Data Acquisition

Use the `get_data.ipynb` notebook to download and prepare the dataset:

```bash
jupyter notebook get_data.ipynb
```

The notebook will:
1. Download the dataset from Kaggle
2. Extract and organize files
3. Save to `data/amex-raddar-parquet/`

---

## 🏗️ Pipeline Architecture

```mermaid
graph LR
    A[Raw Parquet Data] --> B[Data Loading]
    B --> C[Data Cleaning]
    C --> D[Null Handling]
    D --> E[Feature Engineering]
    E --> F[Imputation]
    F --> G[Encoding]
    G --> H[Vectorization]
    H --> I[Scaling]
    I --> J[Train-Test Split]
    J --> K[Class Balancing]
    K --> L[GBT Training]
    L --> M[Model Evaluation]
    M --> N[Predictions]
```

### Pipeline Stages

| Stage | Description | Tools |
|-------|-------------|-------|
| **1. Data Loading** | Load Parquet files using Spark | `spark.read.parquet()` |
| **2. Data Cleaning** | Remove fully-null rows & high-missing columns | Custom filters |
| **3. Null Profiling** | Analyze null/NaN patterns across features | Spark SQL functions |
| **4. Imputation** | Fill missing values (numeric & categorical) | `Imputer`, custom logic |
| **5. Encoding** | Convert categorical variables to numeric | `StringIndexer` |
| **6. Vectorization** | Combine features into vector format | `VectorAssembler` |
| **7. Scaling** | Standardize numeric features | `StandardScaler` |
| **8. Train-Test Split** | Split data for training and validation | `randomSplit()` |
| **9. Class Balancing** | Handle imbalanced classes | Class weights |
| **10. Model Training** | Train Gradient Boosted Trees classifier | `GBTClassifier` |
| **11. Evaluation** | Assess model performance | ROC-AUC, Confusion Matrix |

---

## ✨ Features

### Data Processing
- ✅ Efficient Parquet file handling
- ✅ Distributed data cleaning and preprocessing
- ✅ Intelligent null/NaN handling
- ✅ Automatic feature type detection
- ✅ Optimized partitioning and caching

### Machine Learning
- ✅ Gradient Boosted Trees (GBT) classifier
- ✅ Class imbalance handling with weights
- ✅ Comprehensive feature engineering
- ✅ Scalable model training
- ✅ Multiple evaluation metrics

### Performance
- ✅ Adaptive query execution
- ✅ Kryo serialization for speed
- ✅ Configurable Spark resources (cores, memory)
- ✅ Optimized shuffle partitions

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Java 8 or 11 (for Spark)
- Apache Spark 3.x
- Jupyter Notebook

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AMBehroozi/spark_bigdata_classification.git
   cd spark_bigdata_classification
   ```

2. **Install dependencies**
   ```bash
   pip install pyspark jupyter matplotlib requests
   ```

3. **Configure Spark** (Optional)
   
   Adjust Spark settings in `American_Express.ipynb` based on your system:
   ```python
   SPARK_CORES = 12          # Number of CPU cores
   DRIVER_MEM   = "64g"       # Driver memory
   SHUFFLE_PARTS = 320       # Shuffle partitions
   ```

---

## 💻 Usage

### Step 1: Download Data

Run the data acquisition notebook:

```bash
jupyter notebook get_data.ipynb
```

### Step 2: Run the ML Pipeline

Open and execute the main pipeline notebook:

```bash
jupyter notebook American_Express.ipynb
```

The notebook will:
1. Initialize Spark session with optimized settings
2. Load and cache the Parquet data
3. Perform null/NaN profiling
4. Clean and preprocess features
5. Engineer features and prepare for modeling
6. Train the GBT classifier
7. Evaluate model performance
8. Generate predictions

### Step 3: Review Results

The notebook includes:
- Data quality reports
- Feature statistics
- Model performance metrics (ROC-AUC, confusion matrix)
- Prediction outputs

---

## 🚀 API Deployment

Deploy the trained model as a **REST API** for real-time predictions using FastAPI.

### Quick Start

1. **Train the model**
   ```bash
   python train_model.py
   ```
   This creates the `models/` directory with trained artifacts.

2. **Install API dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API server**
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
      "customer_id": "CUST123",
      "features": {
        "B_1": 0.5,
        "B_2": 1.2,
        "B_30": "A",
        "D_39": 0.8
      }
    }
  }'
```

**Response:**
```json
{
  "customer_id": "CUST123",
  "prediction": 0,
  "probability": 0.23,
  "risk_level": "low"
}
```

#### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"customer_id": "CUST123", "features": {...}},
      {"customer_id": "CUST456", "features": {...}}
    ]
  }'
```

### Docker Deployment

**Build and run with Docker Compose:**

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop the service
docker-compose down
```

The API will be available at `http://localhost:8000`

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "customer": {
            "customer_id": "CUST123",
            "features": {
                "B_1": 0.5,
                "B_2": 1.2,
                # ... more features
            }
        }
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### Production Considerations

- **Authentication**: Add API keys or OAuth for production
- **Rate Limiting**: Implement rate limiting to prevent abuse
- **Monitoring**: Set up logging and metrics (Prometheus, Grafana)
- **Scaling**: Deploy multiple instances behind a load balancer
- **Model Storage**: Use cloud storage (S3, GCS) for model artifacts

📖 **For detailed API documentation**, see [API_USAGE.md](API_USAGE.md)

---

## 📁 Project Structure

```
spark_bigdata_classification/
│
├── American_Express.ipynb    # Main ML pipeline (data cleaning → modeling → evaluation)
├── get_data.ipynb            # Data acquisition script
├── train_model.py            # Standalone model training script
├── app.py                    # FastAPI application
├── schemas.py                # API request/response schemas
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container configuration
├── docker-compose.yml        # Docker orchestration
├── .env.example             # Environment configuration template
├── .gitignore               # Git exclusions
├── README.md                # Project documentation (this file)
├── API_USAGE.md             # Detailed API usage guide
│
├── models/                  # Trained models (created by train_model.py)
│   ├── gbt_credit_default_preprocessing/
│   ├── gbt_credit_default_gbt/
│   └── gbt_credit_default_metadata.json
│
└── data/                    # Data directory (created after running get_data.ipynb)
    └── amex-raddar-parquet/ # Parquet dataset files
        └── data_clean.parquet
```

---

## 🛠️ Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Apache Spark** | Distributed data processing | 3.x |
| **PySpark** | Python API for Spark | 3.x |
| **Spark MLlib** | Machine learning algorithms | 3.x |
| **Parquet** | Columnar storage format | - |
| **Python** | Programming language | 3.8+ |
| **Jupyter** | Interactive development | Latest |
| **Matplotlib** | Visualization | Latest |

### Why These Technologies?

- **Spark**: Handles datasets too large for single-machine processing
- **Parquet**: 10-100x faster than CSV for analytical queries
- **MLlib**: Scalable ML algorithms designed for big data
- **Jupyter**: Interactive exploration and documentation

---

## 📈 Results

The pipeline produces:

- **Model Performance Metrics**
  - ROC-AUC score
  - Confusion matrix
  - Precision, Recall, F1-Score
  
- **Data Quality Reports**
  - Null/NaN percentages per feature
  - Feature distributions
  - Class balance statistics

- **Predictions**
  - Default probability scores
  - Binary classifications

> **Note**: Specific performance metrics depend on the dataset and hyperparameters used.

---

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🔧 Submit pull requests

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**AMBehroozi**

- GitHub: [@AMBehroozi](https://github.com/AMBehroozi)

---

## 🙏 Acknowledgments

- Dataset: American Express Credit Default Dataset (Kaggle)
- Apache Spark community for excellent documentation
- PySpark MLlib for scalable ML tools

---

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">
  <strong>⭐ If you find this project useful, please consider giving it a star! ⭐</strong>
</div>
