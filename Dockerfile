FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY MLTesting/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY MLTesting/ /app/

# Expose port
EXPOSE 8080

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["python", "app.py"]
