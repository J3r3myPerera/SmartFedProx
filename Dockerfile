FROM python:3.11.11-slim

WORKDIR /app

# Install torch CPU wheel first (largest dependency — isolated for better layer caching)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu

# Copy and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE $PORT

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
