@echo off
color a
chcp 65001 >nul

echo ========================================
echo 🔍 МОНИТОРИНГ ПРОГРЕССА ОБРАБОТКИ
echo ========================================
echo.

cd /d "%~dp0"

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
call .venv\Scripts\activate

echo.
echo 🚀 Запуск мониторинга прогресса...
echo 💡 Нажмите Ctrl+C для остановки
echo.

python monitor_progress.py

echo.
echo 👋 Мониторинг завершен
pause 