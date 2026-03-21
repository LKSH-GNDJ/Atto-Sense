@echo off
title AttoSense — Streamlit Frontend
echo.
echo  ╔══════════════════════════════════════╗
echo  ║   AttoSense · Streamlit UI           ║
echo  ║   http://localhost:8501              ║
echo  ╚══════════════════════════════════════╝
echo.
call venv\Scripts\activate
streamlit run frontend\app.py --server.port 8501
pause
