@echo off
title Autonomous Financial Wellness Agent
echo ============================================================
echo   Starting Autonomous Financial Wellness Agent...
echo ============================================================

cd /d "%~dp0"

if not exist ".venv" (
    echo [1/3] Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Python not found or failed to create venv.
        echo Please ensure Python 3.10-3.12 is installed and added to PATH.
        pause
        exit /b 1
    )
    echo [2/3] Installing dependencies from requirements.txt...
    call .venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo [1/2] Activating existing virtual environment...
    call .venv\Scripts\activate.bat
)

echo [2/2] Launching Streamlit App...
echo App will open in your default browser at http://localhost:8501
streamlit run app.py
pause
