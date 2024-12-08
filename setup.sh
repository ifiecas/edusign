#!/bin/bash
pip install --no-cache-dir -r requirements.txt
python -m streamlit run edusign.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
