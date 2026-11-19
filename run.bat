@echo off
title Autonomous Financial Wellness Agent
cd /d "%~dp0"
echo ============================================================
echo   Starting Autonomous Financial Wellness Agent...
echo ============================================================

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [1/3] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Python is not available. Install Python 3.10-3.12 and add it to PATH.
        pause
        exit /b 1
    )
)

"%PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [2/3] Installing project dependencies...
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo [ERROR] Could not upgrade pip.
        pause
        exit /b 1
    )
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install project dependencies.
        pause
        exit /b 1
    )
)

echo [3/3] Launching Streamlit at http://localhost:8501 ...
"%PYTHON%" -m streamlit run app.py
if errorlevel 1 (
    echo [ERROR] The application stopped. See the message above.
    pause
)
