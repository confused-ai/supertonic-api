FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories — supertonic stores model at ~/.cache/supertonic3/
RUN mkdir -p /root/.cache/supertonic3 && \
    mkdir -p /root/.cache/huggingface/hub && \
    chmod -R 777 /root/.cache

# Set environment variables for cache
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_DISABLE_XET=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app directory
COPY app/ ./app/

EXPOSE 8800

# Start with Uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8800} --workers ${WORKERS:-${WEB_CONCURRENCY:-1}}"]
