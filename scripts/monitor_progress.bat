@echo off
color a
chcp 65001 >nul

echo ========================================
echo 🔍 МОНИТОРИНГ ПРОГРЕССА ОБРАБОТКИ
echo ========================================
echo.

REM Получаем путь к директории скрипта
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

REM Переходим в корневую директорию проекта
cd /d "%PROJECT_DIR%"

echo 📁 Проверка виртуального окружения...
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Виртуальное окружение не найдено!
    echo 💡 Запустите setup_project.bat для настройки
    pause
    exit /b 1
)

echo ✅ Виртуальное окружение найдено
echo.

echo 🔄 Активация виртуального окружения...
call ".venv\Scripts\activate.bat"

echo.
echo 🚀 Запуск мониторинга прогресса...
echo 💡 Нажмите Ctrl+C для остановки
echo.

python "monitor_progress.py"

echo.
echo 👋 Мониторинг завершен
pause 