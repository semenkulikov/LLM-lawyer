@echo off
color a
chcp 65001 >nul
echo ========================================
echo    Полный пайплайн обучения модели
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

REM Проверяем наличие PDF файлов
if not exist "data\raw\*.pdf" (
    echo ПРЕДУПРЕЖДЕНИЕ: PDF файлы не найдены в папке data\raw
    echo Поместите PDF документы в папку data\raw перед запуском
    echo.
    set /p choice="Продолжить без данных? (y/n): "
    if /i not "%choice%"=="y" (
        echo Отмена запуска.
        pause
        exit /b 1
    )
)

REM Проверяем настройки
if not exist ".env" (
    echo ПРЕДУПРЕЖДЕНИЕ: Файл .env не найден
    echo Скопируйте env.example в .env и настройте параметры
    echo.
    set /p choice="Продолжить с настройками по умолчанию? (y/n): "
    if /i not "%choice%"=="y" (
        echo Отмена запуска.
        pause
        exit /b 1
    )
)

echo Запуск полного пайплайна обучения...
echo Это может занять несколько часов в зависимости от количества документов
echo.
echo Этапы:
echo 1. Предобработка PDF документов
echo 2. Анализ с помощью OpenAI (если настроен API ключ)
echo 3. Создание обучающего датасета
echo 4. Обучение модели
echo.

REM Запускаем полный пайплайн
python "run_pipeline.py"

echo.
echo Пайплайн завершен.
echo Модель сохранена в папке models\legal_model
echo Теперь можно запустить start_gui.bat для использования модели
pause 