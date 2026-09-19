@echo off
title RetinaScope - AI Screening Platform

cd /d "%~dp0"

echo.
echo ==========================================
echo          RETINASCOPE
echo   AI Screening ^& Resource Planning
echo ==========================================
echo.
echo Starting RetinaScope...
echo Please wait...
echo.

call venv\Scripts\activate.bat

streamlit run app.py

pause