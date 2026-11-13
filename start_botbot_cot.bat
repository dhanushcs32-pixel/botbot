@echo off
title BotBot Secure Vault Launcher (CoT)
cd /d C:\botbot
echo [VAULT] Starting BotBot (CoT) with secure personal_data vault...
echo [INFO] CoT diagnostic UI: C:\botbot\chat-cot.html
powershell -ExecutionPolicy Bypass -File "C:\botbot\vault.ps1" run-bot
pause
