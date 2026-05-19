@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0preview.ps1"
pause
