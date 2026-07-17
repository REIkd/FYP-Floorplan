@echo off
chcp 65001 >nul
echo ================================================================================
echo Ollama - CPU mode (fixes CUDA / GPU driver errors)
echo ================================================================================
echo.
echo Quit Ollama from the system tray first, then press any key to start CPU mode...
pause >nul

taskkill /IM ollama.exe /F >nul 2>&1
taskkill /IM "ollama app.exe" /F >nul 2>&1
timeout /t 2 /nobreak >nul

set CUDA_VISIBLE_DEVICES=-1
set OLLAMA_NUM_GPU=0
set OLLAMA_HOST=127.0.0.1:11434

echo Starting Ollama on CPU (no GPU)...
echo Environment: CUDA_VISIBLE_DEVICES=-1, OLLAMA_NUM_GPU=0
start "Ollama CPU" cmd /c "set CUDA_VISIBLE_DEVICES=-1&& set OLLAMA_NUM_GPU=0&& ollama serve"

timeout /t 3 /nobreak >nul
echo.
echo Ollama should be running at http://localhost:11434
echo Refresh the web app AI Assistant tab and try again.
echo.
echo Note: CPU mode is slower but avoids "device kernel image is invalid" errors.
echo To use GPU again, quit this Ollama and restart from the system tray app.
echo.
pause
