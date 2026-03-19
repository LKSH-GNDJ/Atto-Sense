@echo off
start cmd /k "venv\Scripts\activate && python -m backend.api"
timeout /t 5
start cmd /k "venv\Scripts\activate && python -m streamlit run frontend/app.py --server.headless true"