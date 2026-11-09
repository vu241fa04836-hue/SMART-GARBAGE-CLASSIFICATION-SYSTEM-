@echo off
title Smart Garbage Classification System
color 0A

echo =====================================================================
echo    SMART GARBAGE CLASSIFICATION SYSTEM USING AI AND MACHINE LEARNING
echo =====================================================================
echo.
echo Starting Flask Web Application Server...
echo.

where py >nul 2>nul
if %errorlevel% equ 0 (
    py app.py
    goto end
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    python app.py
    goto end
)

if exist "C:\Users\Dell\AppData\Local\Programs\Python\Python313\python.exe" (
    "C:\Users\Dell\AppData\Local\Programs\Python\Python313\python.exe" app.py
    goto end
)

echo [ERROR] Python not found in PATH or standard directory!
pause

:end
