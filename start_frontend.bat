@echo off
echo Starting BotTrain Frontend...
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run frontend/app.py
pause
