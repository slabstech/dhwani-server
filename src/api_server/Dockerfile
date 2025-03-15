# Use official Python runtime as base image
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser
# Expose port from settings (7860 from your code)
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:7860/v1/health || exit 1

# Command to run the application with configurable host/port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]