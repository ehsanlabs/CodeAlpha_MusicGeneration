FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for music21
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Hugging Face Spaces requires port 7860
EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "180", "--workers", "1", "web_app:app"]
