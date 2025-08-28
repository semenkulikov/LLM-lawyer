@echo off
color a
chcp 65001 >nul
echo ========================================
echo    Быстрый тест системы LLM-lawyer
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

echo Запуск диагностики системы...
echo.

REM Проверяем наличие .env файла
if not exist ".env" (
    echo ПРЕДУПРЕЖДЕНИЕ: Файл .env не найден!
    echo Скопируйте env.example в .env и настройте API ключи
    echo.
    set /p choice="Продолжить без .env файла? (y/n): "
    if /i not "%choice%"=="y" (
        echo Отмена запуска.
        pause
        exit /b 1
    )
)

REM Запускаем быстрый тест
python "quick_test.py"

echo.
echo Диагностика завершена.
echo Проверьте результаты выше.
pause 