FROM ubuntu:22.04

WORKDIR /app

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10
RUN apt-get install -y python3-pip


# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/app/config/.env



