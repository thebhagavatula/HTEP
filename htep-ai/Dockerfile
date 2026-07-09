FROM python:3.11-slim

# System deps for OpenCV, pdf2image, Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies first (better Docker cache)
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Download spaCy model at build time
RUN python -m spacy download en_core_web_sm

# Pre-download PaddleOCR models at build time so the container starts instantly
# (Without this, Cloud Run will timeout while downloading them on every cold start)
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)"

# Copy project files
COPY --chown=user . .

# Ensure data directories exist
RUN mkdir -p data/raw data/processed data/dictionaries

# Cloud Run injects the PORT environment variable (default 8080)
# Make Python logs unbuffered so they show up immediately in Cloud Logging
ENV PYTHONUNBUFFERED True

# Run gunicorn using exec, bind directly to $PORT, and disable gunicorn's internal timeout (Cloud Run handles timeouts)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 0 app.api:app
