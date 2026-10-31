@echo off
REM ============================================
REM  Botbot Launcher (CoT diagnostic)
REM  Starts the SAME NPU server + main API server
REM  as start_botbot.bat — the CoT endpoint lives
REM  inside main.py, not a separate process.
REM  Window titles are suffixed (CoT) so you can
REM  tell at a glance which set is running.
REM ============================================

set PROJECT_DIR=C:\botbot

start "NPU Server (CoT)" cmd /k "cd /d %PROJECT_DIR% && .\venv\Scripts\python.exe npu_server.py"

REM Small delay so the NPU server has a moment to start before main.py hits it
timeout /t 2 /nobreak >nul

start "Botbot Server (CoT)" cmd /k "cd /d %PROJECT_DIR% && .\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo.
echo ============================================
echo  Both servers started (CoT launcher).
echo  When they are up, open this file manually
echo  in your browser to use the CoT diagnostic:
echo.
echo  C:\botbot\chat-cot.html
echo.
echo  (POSTs go to /chat/cot with the activation
echo   header; requires the primary API key.)
echo ============================================
