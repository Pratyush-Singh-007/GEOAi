FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Use USE_SLM=false for CPU-only containers without GPU
ENV USE_SLM=false

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
