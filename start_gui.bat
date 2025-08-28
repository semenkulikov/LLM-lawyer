@echo off
color a
chcp 65001 >nul
echo ========================================
echo    Графический интерфейс LLM-lawyer
echo ========================================
echo.

REM Переходим в папку проекта
cd /d "%~dp0"

REM Активируем виртуальное окружение
echo Активация виртуального окружения...
call .venv\Scripts\activate

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

REM Проверяем наличие Gemini GUI
if exist "gui\gemini_hybrid_app.py" (
    echo Найдена гибридная система с Gemini Ultra
    echo.
    set /p choice="Запустить гибридный GUI с Gemini? (y/n): "
    if /i "%choice%"=="y" (
        echo Запуск гибридного GUI с Gemini Ultra...
        python gui\gemini_hybrid_app.py
        goto :end
    )
)

echo Запуск обычного графического интерфейса...
echo Откроется окно с интерфейсом
echo Для остановки закройте окно браузера и нажмите Ctrl+C в этом окне
echo.

REM Запускаем обычный GUI
python gui\app.py

:end
echo.
echo Графический интерфейс остановлен.
pause 