@echo off
echo ========================================
echo IMARA AI Disease Detection Service
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo.

REM Check if requirements are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Check if .env exists
if not exist ".env" (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo.
    echo ========================================
    echo IMPORTANT: Edit .env file and set your API_KEY!
    echo ========================================
    echo.
    pause
)

REM Check if model exists
if not exist "models\plantvillage_mobilenetv2.h5" (
    echo ========================================
    echo WARNING: Model file not found!
    echo ========================================
    echo.
    echo The AI service will run in DEMO MODE.
    echo.
    echo To use the real model:
    echo 1. Create 'models' folder
    echo 2. Download PlantVillage MobileNetV2 model from:
    echo    https://www.kaggle.com/datasets/emmarex/plantdisease
    echo 3. Place model at: models\plantvillage_mobilenetv2.h5
    echo.
    echo Press any key to start in DEMO MODE...
    pause >nul
    echo.
)

echo Starting AI Service...
echo Service will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the service
echo.

python main.py
