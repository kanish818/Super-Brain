FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=webapp.py
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Run the application
CMD gunicorn webapp:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
