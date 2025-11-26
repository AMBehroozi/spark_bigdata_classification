FROM python:3.10-slim

# Install Java (required for Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set Java home
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY schemas.py .
COPY train_model.py .

# Copy models (if they exist)
# Note: In production, models should be mounted as volumes or loaded from cloud storage
COPY models/ ./models/ 2>/dev/null || true

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=models
ENV SPARK_DRIVER_MEMORY=4g
ENV SPARK_EXECUTOR_MEMORY=4g

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
