# ML Training Dashboard - Dockerfile
# Optimized for Railway.com with 32GB RAM / 32 CPU

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Railway environment detection (for DataLoader worker config)
ENV RAILWAY_ENVIRONMENT=production

# PyTorch optimization for multi-CPU
ENV OMP_NUM_THREADS=32
ENV MKL_NUM_THREADS=32
ENV NUMEXPR_NUM_THREADS=32
ENV OPENBLAS_NUM_THREADS=32

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data_storage/raw data_storage/processed data_storage/models logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Run the web server
# Railway provides $PORT env var, fallback to 8000 for local
CMD python -m uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}
