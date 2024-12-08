#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Install Python dependencies
echo "Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# Start the Streamlit application
echo "Starting Streamlit app..."
exec python -m streamlit run edusign.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
