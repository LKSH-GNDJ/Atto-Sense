@echo off
title AttoSense — CLI Bot
echo.
echo  ╔══════════════════════════════════════╗
echo  ║   AttoSense · CLI Bot                ║
echo  ║   Universal Intent Classifier        ║
echo  ╚══════════════════════════════════════╝
echo.
call venv\Scripts\activate
python bot.py %*
pause
