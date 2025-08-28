@echo off
chcp 65001 >nul
echo.
echo ========================================
echo    Обучение модели с LoRA
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
cd /d "%PROJECT_DIR%"

echo Активация виртуального окружения...
call .venv\Scripts\activate.bat

echo.
echo Проверка конфигурации...
if not exist "datasets\lora_training_config.json" (
    echo ❌ Конфигурация не найдена! Сначала запустите очистку датасета.
    echo.
    pause
    exit /b 1
)

echo.
echo Начинаю обучение с LoRA...
python src/train_lora.py --config datasets/lora_training_config.json

echo.
echo Обучение завершено!
echo.
pause
