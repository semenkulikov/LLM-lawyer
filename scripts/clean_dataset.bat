@echo off
chcp 65001 >nul
echo.
echo ========================================
echo    Очистка датасета для обучения LoRA
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
cd /d "%PROJECT_DIR%"

echo Активация виртуального окружения...
call .venv\Scripts\activate.bat

echo.
echo Очистка датасета...
python src/dataset_cleaner.py

echo.
echo Очистка завершена!
echo.
pause
