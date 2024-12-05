#!/bin/bash
apt-get update
apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
python -m streamlit run edusign.py --server.port 8000 --server.address 0.0.0.0
