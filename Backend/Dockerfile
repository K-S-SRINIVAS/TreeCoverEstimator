# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run Flask app (change if using FastAPI or uvicorn)
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]

