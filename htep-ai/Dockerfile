FROM python:3.11-slim

# System deps for OpenCV, pdf2image, Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better Docker cache)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Download spaCy model at build time
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Ensure data directories exist
RUN mkdir -p data/raw data/processed data/dictionaries

EXPOSE 5000

# Use gunicorn for production (120s timeout for long OCR jobs)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "--threads", "2", "app.api:app"]
