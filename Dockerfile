# Use a lightweight version of Python
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies (with increased timeout & retries)
COPY requirements.txt .
RUN pip install \
      --default-timeout=100 \
      --retries=5 \
      --no-cache-dir \
      -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
