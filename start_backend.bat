@echo off
title AttoSense — Backend
echo.
echo  ╔══════════════════════════════════════╗
echo  ║   AttoSense · Backend                ║
echo  ║   http://localhost:8000              ║
echo  ║   http://localhost:8000/docs         ║
echo  ╚══════════════════════════════════════╝
echo.
call venv\Scripts\activate
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000 --log-level info
pause
