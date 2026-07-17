@echo off
chcp 65001 >nul
cls
echo ================================================================================
echo Floor Plan Analyzer - Web Application
echo ================================================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup_venv.bat first to create the project environment.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Starting web application...
echo.
echo The app will open in your default browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

streamlit run app.py

pause

