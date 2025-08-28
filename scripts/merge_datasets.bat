@echo off
chcp 65001 >nul
echo.
echo ========================================
echo    Объединение датасетов для обучения
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
cd /d "%PROJECT_DIR%"

echo Активация виртуального окружения...
call .venv\Scripts\activate.bat

echo.
echo Объединение датасетов...
python src/merge_datasets.py

echo.
echo Объединение завершено!
echo.
pause
