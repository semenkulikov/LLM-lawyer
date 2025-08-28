@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   Юридический ассистент - GUI запуск
echo ========================================
echo.

REM Активация виртуального окружения
if exist "venv\Scripts\activate.bat" (
    echo Активация виртуального окружения...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Активация виртуального окружения...
    call .venv\Scripts\activate.bat
) else (
    echo Виртуальное окружение не найдено!
    echo Убедитесь, что оно создано и активировано.
    pause
    exit /b 1
)

echo.
echo Проверка зависимостей...
python -c "import tkinter" 2>nul
if errorlevel 1 (
    echo Ошибка: tkinter не установлен!
    echo Установите Python с поддержкой tkinter.
    pause
    exit /b 1
)

echo.
echo Проверка модели...
if not exist "models\legal_model" (
    echo ВНИМАНИЕ: Модель не найдена!
    echo Сначала обучите модель командой:
    echo python src\train.py --train_file data\train_dataset.jsonl --output_dir models\legal_model
    echo.
    set /p choice="Продолжить без модели? (y/n): "
    if /i not "%choice%"=="y" (
        echo Запуск отменен.
        pause
        exit /b 1
    )
)

echo.
echo Запуск современного универсального гибридного GUI...
echo Откроется окно с современным интерфейсом и анимациями
echo Для остановки закройте окно и нажмите Ctrl+C в этом окне
echo.

REM Запускаем исправленный универсальный гибридный GUI
python gui\fixed_hybrid_app.py

echo.
echo Графический интерфейс остановлен.
pause 