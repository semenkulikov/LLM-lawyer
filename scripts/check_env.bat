@echo off
color a
chcp 65001 >nul
echo ========================================
echo    Проверка конфигурации .env
echo ========================================
echo.

REM Получаем путь к директории скрипта
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

REM Переходим в корневую директорию проекта
cd /d "%PROJECT_DIR%"

REM Активируем виртуальное окружение
echo Активация виртуального окружения...
call ".venv\Scripts\activate.bat"

REM Проверяем, что окружение активировано
if not defined VIRTUAL_ENV (
    echo ОШИБКА: Не удалось активировать виртуальное окружение!
    echo Убедитесь, что проект правильно установлен.
    pause
    exit /b 1
)

echo Виртуальное окружение активировано.
echo.

echo Запуск проверки .env файла...
echo.

REM Запускаем проверку
python "check_env.py"

echo.
echo Проверка завершена.
pause 