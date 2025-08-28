@echo off
color a
chcp 65001 >nul
echo ========================================
echo    Автоматическая настройка проекта
echo ========================================
echo.

REM Получаем путь к директории скрипта
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

REM Переходим в корневую директорию проекта
cd /d "%PROJECT_DIR%"

echo Проверка системы...
echo.

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    echo Установите Python 3.11 с сайта python.org
    pause
    exit /b 1
)

echo Python найден.
echo.

REM Проверяем наличие Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Git не найден!
    echo Установите Git с сайта git-scm.com
    pause
    exit /b 1
)

echo Git найден.
echo.

REM Создаем виртуальное окружение
echo Создание виртуального окружения...
if exist ".venv" (
    echo Виртуальное окружение уже существует.
) else (
    python -m venv .venv
    echo Виртуальное окружение создано.
)

REM Активируем виртуальное окружение
echo Активация виртуального окружения...
call ".venv\Scripts\activate.bat"

REM Проверяем, что окружение активировано
if not defined VIRTUAL_ENV (
    echo ОШИБКА: Не удалось активировать виртуальное окружение!
    pause
    exit /b 1
)

echo Виртуальное окружение активировано.
echo.

REM Обновляем pip
echo Обновление pip...
python -m pip install --upgrade pip

REM Устанавливаем PyTorch с CUDA
echo Установка PyTorch с CUDA поддержкой...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Устанавливаем остальные зависимости
echo Установка зависимостей...
pip install -r requirements.txt

echo.
echo Создание структуры папок...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\structured" mkdir data\structured
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "results" mkdir results

echo Папки созданы.
echo.

REM Создаем .env файл если его нет
if not exist ".env" (
    echo Создание файла конфигурации...
    copy env.example .env
    echo Файл .env создан. 
    echo.
    echo ВАЖНО: Отредактируйте файл .env и добавьте ваш OpenAI API ключ!
    echo Замените строку: OPENAI_API_KEY=sk-your-openai-api-key-here
    echo На ваш реальный API ключ.
    echo.
)

echo.
echo ========================================
echo    Настройка завершена!
echo ========================================
echo.
echo Следующие шаги:
echo 1. Отредактируйте файл .env и добавьте OpenAI API ключ
echo 2. Поместите PDF документы в папку data\raw
echo 3. Запустите run_pipeline.bat для обучения модели
echo 4. Запустите start_gui.bat для использования модели
echo.
echo Для проверки системы запустите quick_test.bat
echo.
pause 