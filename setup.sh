#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Inside setup.sh
echo "Creating virtual environment..."
python -m venv antenv
source antenv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt


# Install Python dependencies
echo "Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# Start the Streamlit application
echo "Starting Streamlit app..."
exec python -m streamlit run edusign.py --server.port $PORT --server.address 0.0.0.0 --server.headless true


