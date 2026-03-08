@echo off
echo Starting BotTrain Backend...
cd /d "%~dp0"
call venv\Scripts\activate.bat
uvicorn backend.api:app --reload
pause
