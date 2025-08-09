# Руководство по развертыванию LLM-lawyer

## Системные требования

### Минимальные требования:
- **ОС**: Windows 10/11 (64-bit)
- **RAM**: 16 GB (рекомендуется 32 GB для больших датасетов)
- **GPU**: NVIDIA с поддержкой CUDA 11.8+ (рекомендуется RTX 3060 или выше)
- **Диск**: 50 GB свободного места
- **Интернет**: Стабильное подключение для загрузки моделей

### Рекомендуемые требования:
- **RAM**: 32 GB
- **GPU**: RTX 4070/4080/4090 или A100/H100
- **Диск**: SSD 500 GB+
- **CPU**: Intel i7/AMD Ryzen 7 или выше

## Пошаговое развертывание

### Шаг 1: Установка базового ПО

#### 1.1 Установка Python
```bash
# Скачать Python 3.11 с официального сайта
# https://www.python.org/downloads/release/python-3118/
# Выбрать "Windows installer (64-bit)"
# При установке ОБЯЗАТЕЛЬНО поставить галочку "Add Python to PATH"
```

#### 1.2 Установка Git
```bash
# Скачать Git с официального сайта
# https://git-scm.com/download/win
# Использовать настройки по умолчанию
```

#### 1.3 Установка CUDA Toolkit
```bash
# Скачать CUDA Toolkit 12.1 с сайта NVIDIA
# https://developer.nvidia.com/cuda-12-1-0-download-archive
# Выбрать Windows → x86_64 → 10/11 → exe (local)
# Установить с настройками по умолчанию
```

### Шаг 2: Клонирование проекта

```bash
# Открыть командную строку (cmd) от имени администратора
# Перейти в папку, где будет проект (например, C:\Projects)
cd C:\Projects

# Клонировать репозиторий
git clone https://github.com/semenkulikov/LLM-lawyer.git

# Перейти в папку проекта
cd LLM-lawyer
```

### Шаг 3: Создание виртуального окружения

```bash
# Создать виртуальное окружение
python -m venv .venv

# Активировать виртуальное окружение
.venv\Scripts\activate

# Проверить, что окружение активировано (должен появиться (.venv) в начале строки)
```

### Шаг 4: Установка зависимостей

```bash
# Обновить pip
python -m pip install --upgrade pip

# Установить PyTorch с CUDA поддержкой
pip install -r requirements-torch.txt

# Установить остальные зависимости
pip install -r requirements.txt

# Проверить установку CUDA
python check_cuda.py
```

### Шаг 5: Настройка конфигурации

#### 5.1 Создание файла .env
```bash
# Скопировать пример конфигурации
copy env.example .env

# Отредактировать .env файл (открыть в блокноте)
notepad .env
```

**Содержимое .env файла:**
```env
# OpenAI API ключ (получить на https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Настройки для больших датасетов
MAX_DOCS=1000
MAX_WORKERS=5
BATCH_SIZE=4
LEARNING_RATE=5e-5

# Настройки модели
MODEL_NAME=Qwen/Qwen1.5-3B-Chat
MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=1024

# Настройки логирования
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Настройки мониторинга
TENSORBOARD_PORT=6006
WANDB_PROJECT=llm-lawyer
```

#### 5.2 Создание структуры папок
```bash
# Создать необходимые папки
mkdir data\raw
mkdir data\processed
mkdir data\structured
mkdir models
mkdir logs
mkdir results
```

### Шаг 6: Подготовка данных

#### 6.1 Размещение PDF документов
```bash
# Скопировать все PDF документы в папку data\raw
# Например:
copy "C:\Users\Username\Documents\Legal_Documents\*.pdf" data\raw\
```

#### 6.2 Проверка данных
```bash
# Проверить количество файлов
dir data\raw\*.pdf

# Запустить быстрый тест системы
python quick_test.py
```

### Шаг 7: Запуск полного пайплайна

#### 7.1 Первый запуск (обучение с нуля)
```bash
# Запустить полный пайплайн
python run_pipeline.py

# Или пошагово:

# 0. Запуск мониторинга
python monitor_progress.py

# 1. Предобработка
python src\preprocess.py --input-dir data\raw --output-dir data\processed

# 2. Анализ с OpenAI (если есть API ключ)
python src/process_with_openai.py --input-dir data/processed --output-dir data/analyzed --max-docs 3 --max-workers 1

# 3. Создание датасета
python src\build_dataset.py --input-dir data\structured --output-file data\train_dataset.jsonl

# 4. Обучение модели
python src\train.py --train_file data\train_dataset.jsonl --output_dir models\legal_model --epochs 3 --batch_size 4
```

#### 7.2 Тестирование модели
```bash
# Создать тестовый файл
echo "Факты дела: Истец обратился в суд с требованием о взыскании задолженности по договору займа в размере 100000 рублей." > test_input.txt

# Запустить генерацию
python src\inference.py --model_path models\legal_model --input_file test_input.txt --output_file test_output.txt

# Проверить результат
type test_output.txt
```

### Шаг 8: Настройка автоматического дообучения

#### 8.1 Запуск мониторинга новых документов
```bash
# Запустить автоматическое дообучение в фоновом режиме
python src\auto_incremental.py --watch
```

#### 8.2 Настройка планировщика задач (опционально)
```bash
# Создать задачу в планировщике Windows для автоматического запуска
# Путь к скрипту: C:\Projects\LLM-lawyer\start_auto_training.bat
```

### Шаг 9: Запуск GUI

```bash
# Запустить графический интерфейс
python gui\app.py
```

## Батники для заказчика

### start_auto_training.bat
```batch
@echo off
cd /d "C:\Projects\LLM-lawyer"
call .venv\Scripts\activate
python src\auto_incremental.py --watch
pause
```

### start_gui.bat
```batch
@echo off
cd /d "C:\Projects\LLM-lawyer"
call .venv\Scripts\activate
python gui\app.py
pause
```

### quick_test.bat
```batch
@echo off
cd /d "C:\Projects\LLM-lawyer"
call .venv\Scripts\activate
python quick_test.py
pause
```

### run_pipeline.bat
```batch
@echo off
cd /d "C:\Projects\LLM-lawyer"
call .venv\Scripts\activate
python run_pipeline.py
pause
```

## Мониторинг и отладка

### Просмотр логов
```bash
# Просмотр логов в реальном времени
tail -f logs\app.log

# Или в Windows:
type logs\app.log
```

### Мониторинг через TensorBoard
```bash
# Запуск TensorBoard
python src\monitor_training.py --log_dir models --port 6006 --open_browser
```

### Проверка GPU
```bash
# Проверка CUDA
python check_cuda.py

# Проверка использования GPU
nvidia-smi
```

## Устранение неполадок

### Проблема: "CUDA недоступна"
**Решение:**
1. Проверить установку CUDA: `nvidia-smi`
2. Переустановить PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. Проверить версию Python (должна быть 3.11)

### Проблема: "No module named 'pymupdf'"
**Решение:**
```bash
pip install pymupdf
```

### Проблема: "OpenAI API key not found"
**Решение:**
1. Проверить файл `.env`
2. Убедиться, что API ключ корректный
3. Проверить подключение к интернету

### Проблема: "Out of memory"
**Решение:**
1. Уменьшить batch_size в настройках
2. Закрыть другие приложения
3. Использовать градиентное накопление

## Рекомендации по производительности

### Для больших датасетов (50,000+ документов):
1. Использовать `run_large_scale.py` вместо `run_pipeline.py`
2. Настроить `MAX_WORKERS=8` в `.env`
3. Использовать SSD для хранения данных
4. Мониторить температуру GPU

### Оптимизация памяти:
1. Установить `BATCH_SIZE=2` для слабых GPU
2. Использовать градиентное накопление
3. Включить mixed precision training

## Контакты для поддержки

При возникновении проблем:
1. Проверить логи в папке `logs\`
2. Запустить `python quick_test.py`
3. Обратиться к разработчику с описанием ошибки и логами

## Обновление проекта

```bash
# Получить последние изменения
git pull origin main

# Обновить зависимости
pip install -r requirements.txt

# Перезапустить систему
```

---

**Важно:** Всегда делайте резервные копии обученных моделей перед обновлением! 