FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default: run the full pipeline in sequence
# Override with: docker compose run --rm app python src/<script>.py
CMD ["bash", "-c", \
     "python src/edf_to_csv.py && \
      python src/preprocess.py && \
      python src/feature_extraction.py && \
      python src/train_models.py && \
      python src/hmm_smoother.py && \
      python src/evaluate.py && \
      python src/cross_dataset.py"]
