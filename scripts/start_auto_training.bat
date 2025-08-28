@echo off
color a
chcp 65001 >nul
echo ========================================
echo    Автоматическое дообучение модели
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

REM Проверяем наличие модели
if not exist "models\legal_model" (
    echo ПРЕДУПРЕЖДЕНИЕ: Модель не найдена в папке models\legal_model
    echo Сначала необходимо обучить модель с помощью run_pipeline.bat
    echo.
    set /p choice="Продолжить без модели? (y/n): "
    if /i not "%choice%"=="y" (
        echo Отмена запуска.
        pause
        exit /b 1
    )
)

echo Запуск автоматического мониторинга...
echo Система будет отслеживать новые PDF файлы в папке data\raw
echo Для остановки нажмите Ctrl+C
echo.

REM Запускаем автоматическое дообучение
python "src\auto_incremental.py" --watch

echo.
echo Мониторинг остановлен.
pause 