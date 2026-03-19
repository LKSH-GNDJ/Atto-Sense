@echo off
echo Starting AttoSense Frontend...
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run frontend/app.py
pause
