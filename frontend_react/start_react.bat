@echo off
title AttoSense — React UI
echo.
echo  ╔══════════════════════════════════════╗
echo  ║   AttoSense · React UI               ║
echo  ║   http://localhost:3000              ║
echo  ╚══════════════════════════════════════╝
echo.
if not exist node_modules (
    echo  Installing npm packages ^(first run only^)...
    npm install
    echo.
)
npm run dev
pause
