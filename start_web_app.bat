@echo off
chcp 65001 >nul
cls
echo ================================================================================
echo Floor Plan Analyzer - Web Application
echo ================================================================================
echo.
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

