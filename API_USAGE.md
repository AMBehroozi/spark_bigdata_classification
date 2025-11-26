# API Usage Guide

Complete guide for using the Credit Default Prediction API.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will create the `models/` directory with trained artifacts.

### 3. Start the API

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### 4. View API Documentation

Open your browser and navigate to:
- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

---

## API Endpoints

### Health Check

**GET /** or **GET /health**

Check if the API is running and models are loaded.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "spark_active": true,
  "version": "1.0.0"
}
```

---

### Model Information

**GET /model/info**

Get model metadata and performance metrics.

```bash
curl http://localhost:8000/model/info
```

Response:
```json
{
  "model_name": "gbt_credit_default",
  "version": "1.0.0",
  "created_at": "2025-11-26T14:00:00",
  "metrics": {
    "auc": 0.85,
    "precision": 0.75,
    "recall": 0.70,
    "f1_score": 0.72
  },
  "feature_count": 190,
  "categorical_features": ["B_30", "B_38", ...],
  "numeric_features_count": 179
}
```

---

### Single Prediction

**POST /predict**

Make a prediction for a single customer.

**Request:**
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
        "D_39": 0.8,
        "D_114": "B"
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

**Fields:**
- `prediction`: 0 (no default) or 1 (default)
- `probability`: Probability of default (0.0 to 1.0)
- `risk_level`: "low" (<0.3), "medium" (0.3-0.7), or "high" (>0.7)

---

### Batch Prediction

**POST /predict/batch**

Make predictions for multiple customers (up to 1000).

**Request:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "customer_id": "CUST123",
        "features": {"B_1": 0.5, "B_2": 1.2, ...}
      },
      {
        "customer_id": "CUST456",
        "features": {"B_1": 0.8, "B_2": 0.9, ...}
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_id": "CUST123",
      "prediction": 0,
      "probability": 0.23,
      "risk_level": "low"
    },
    {
      "customer_id": "CUST456",
      "prediction": 1,
      "probability": 0.78,
      "risk_level": "high"
    }
  ],
  "total_processed": 2
}
```

---

## Client Examples

### Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "customer": {
            "customer_id": "CUST123",
            "features": {
                "B_1": 0.5,
                "B_2": 1.2,
                "B_30": "A",
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

### JavaScript

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    customer: {
      customer_id: 'CUST123',
      features: {
        B_1: 0.5,
        B_2: 1.2,
        B_30: 'A',
        // ... more features
      }
    }
  })
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data.prediction);
  console.log('Probability:', data.probability);
  console.log('Risk Level:', data.risk_level);
});
```

---

## Docker Deployment

### Build and Run

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

### Access the API

Once running, the API is available at `http://localhost:8000`

---

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid input)
- **422**: Validation Error (schema mismatch)
- **500**: Internal Server Error
- **503**: Service Unavailable (model not loaded)

**Error Response Format:**
```json
{
  "error": "ValidationError",
  "message": "Invalid input data",
  "detail": "Field 'features' is required"
}
```

---

## Performance Considerations

### Batch Processing
- Use `/predict/batch` for multiple predictions
- Maximum batch size: 1000 customers
- More efficient than multiple single requests

### Response Times
- Single prediction: ~100-500ms
- Batch (100 customers): ~1-3s
- First request may be slower (model initialization)

### Rate Limiting
- Not implemented by default
- Consider adding rate limiting for production
- Use tools like `slowapi` or API gateway

---

## Production Deployment

### Recommendations

1. **Use a reverse proxy** (nginx, Traefik)
2. **Add authentication** (API keys, OAuth)
3. **Implement rate limiting**
4. **Set up monitoring** (Prometheus, Grafana)
5. **Use cloud storage for models** (S3, GCS)
6. **Scale horizontally** (multiple API instances)
7. **Configure CORS** appropriately

### Environment Variables

```bash
MODEL_NAME=gbt_credit_default
MODEL_DIR=/path/to/models
SPARK_DRIVER_MEMORY=8g
SPARK_EXECUTOR_MEMORY=8g
LOG_LEVEL=INFO
```

---

## Troubleshooting

### Model Not Found
```
FileNotFoundError: Preprocessing model not found
```
**Solution**: Run `python train_model.py` first to create models.

### Spark Memory Error
```
OutOfMemoryError: Java heap space
```
**Solution**: Increase `SPARK_DRIVER_MEMORY` in environment variables.

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Change port with `uvicorn app:app --port 8001`

---

## Support

For issues or questions:
1. Check the [README](README.md)
2. Review API docs at `/docs`
3. Open an issue on GitHub
