@echo off
REM ============================================
REM  Botbot Launcher
REM  Starts NPU server + main API server in two
REM  separate visible terminal windows.
REM ============================================

set PROJECT_DIR=C:\botbot

start "NPU Server" cmd /k "cd /d %PROJECT_DIR% && .\venv\Scripts\python.exe npu_server.py"

REM Small delay so the NPU server has a moment to start before main.py hits it
timeout /t 2 /nobreak >nul

start "Botbot Server" cmd /k "cd /d %PROJECT_DIR% && .\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000"
