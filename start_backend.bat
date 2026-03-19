@echo off
echo Starting AttoSense Backend...
cd /d "%~dp0"
call venv\Scripts\activate.bat
uvicorn backend.api:app --reload
pause
