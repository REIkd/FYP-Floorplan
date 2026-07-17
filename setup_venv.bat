@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ================================================================================
echo Floor Plan Analyzer - Setup Virtual Environment
echo ================================================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists, updating packages...
)

call venv\Scripts\activate.bat

set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
set NO_PROXY=*

echo.
echo Installing dependencies (this may take several minutes)...
python -m pip install --upgrade pip --proxy=""
pip install --proxy="" -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed. Check your network connection and try again.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Setup complete! Run start_web_app.bat to launch the application.
echo ================================================================================
echo.
pause
