@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoExit -Command "Set-Location -LiteralPath '%SCRIPT_DIR%'; python task.py"
endlocal
