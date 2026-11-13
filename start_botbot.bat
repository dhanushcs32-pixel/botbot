@echo off
title BotBot Secure Vault Launcher
cd /d C:\botbot
echo [VAULT] Starting BotBot with secure personal_data vault...
powershell -ExecutionPolicy Bypass -File "C:\botbot\vault.ps1" run-bot
pause
