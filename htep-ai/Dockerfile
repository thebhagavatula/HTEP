FROM python:3.12-slim

# poppler-utils renders PDF pages for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies first (better Docker cache)
# constraints-deploy.txt pins the full dependency tree to the last known-good build
COPY --chown=user requirements-deploy.txt constraints-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt -c constraints-deploy.txt

# Copy project files
COPY --chown=user . .

# Download the OCR models at build time so a cold start doesn't fetch them
RUN python -c "from src.ocr.extractor import OCRExtractor; OCRExtractor()"

# Ensure data directories exist
RUN mkdir -p data/raw data/processed data/dictionaries

# Make Python logs unbuffered so they show up immediately in the platform's logs
ENV PYTHONUNBUFFERED=1

# The platform injects PORT (Cloud Run: 8080). Jobs run in background threads,
# so a few threads let /jobs/<id> polls through while a document is processed.
CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 4 --timeout 0 app.api:app
