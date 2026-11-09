@echo off
title Train Garbage Classification CNN Model
color 0E

echo =====================================================================
echo    TRAIN CNN GARBAGE CLASSIFIER MODEL (TENSORFLOW / KERAS)
echo =====================================================================
echo.

if exist "C:\Users\Dell\AppData\Local\Programs\Python\Python313\python.exe" (
    "C:\Users\Dell\AppData\Local\Programs\Python\Python313\python.exe" train_model.py
    pause
    goto end
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    python train_model.py
    pause
    goto end
)

echo [ERROR] Python not found!
pause

:end
