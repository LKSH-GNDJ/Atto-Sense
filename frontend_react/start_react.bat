@echo off
echo Installing dependencies (first run only)...
call npm install
echo.
echo Starting AttoSense React UI on http://localhost:3000
echo Make sure the FastAPI backend is running on http://localhost:8000
echo.
call npm run dev
pause
