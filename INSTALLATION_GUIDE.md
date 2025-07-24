# 🚀 Руководство по установке и запуску системы

## 📋 Требования к системе

### Минимальные требования:
- **ОС**: Windows 10/11, Linux, macOS
- **Python**: 3.8 или выше
- **RAM**: 16 GB (рекомендуется 32 GB)
- **GPU**: NVIDIA с 8+ GB VRAM (рекомендуется RTX 3080/4080/4090)
- **Место на диске**: 50 GB свободного места
- **Интернет**: для загрузки моделей и API запросов

### Рекомендуемые требования:
- **ОС**: Windows 11 или Ubuntu 20.04+
- **Python**: 3.10 или выше
- **RAM**: 64 GB
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM)
- **Место на диске**: 100 GB SSD
- **Интернет**: стабильное соединение

## 🔧 Быстрая установка (рекомендуется)

### Шаг 1: Подготовка системы
```bash
# 1. Скачайте и установите Python 3.10+ с официального сайта
# https://www.python.org/downloads/

# 2. Скачайте и установите Git
# https://git-scm.com/downloads

# 3. Скачайте и установите CUDA Toolkit 11.8+ (если есть NVIDIA GPU)
# https://developer.nvidia.com/cuda-downloads
```

### Шаг 2: Клонирование проекта
```bash
# Откройте командную строку (cmd) или PowerShell
git clone https://github.com/your-repo/LLM-lawyer.git
cd LLM-lawyer
```

### Шаг 3: Создание виртуального окружения
```bash
# Создание виртуального окружения
python -m venv .venv

# Активация виртуального окружения
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### Шаг 4: Установка зависимостей
```bash
# Обновление pip
python -m pip install --upgrade pip

# Установка зависимостей
pip install -r requirements.txt
```

### Шаг 5: Настройка OpenAI API
```bash
# Установка переменной окружения
# Windows:
set OPENAI_API_KEY=your_api_key_here
# Linux/macOS:
export OPENAI_API_KEY=your_api_key_here

# Или создайте файл .env в корне проекта:
echo OPENAI_API_KEY=your_api_key_here > .env
```

### Шаг 6: Быстрый запуск
```bash
# Запуск интерактивного меню
python quick_start.py
```

## 🎯 Пошаговая инструкция для заказчика

### Вариант 1: Автоматическая установка (рекомендуется)

1. **Скачайте архив проекта** и распакуйте в удобное место
2. **Откройте командную строку** в папке проекта
3. **Запустите автоматическую установку**:
   ```bash
   python quick_start.py
   ```
4. **Выберите опцию 1** - "Первоначальная настройка системы"
5. **Введите ваш OpenAI API ключ** когда система запросит
6. **Дождитесь завершения установки**

### Вариант 2: Ручная установка

#### 1. Установка Python
- Скачайте Python 3.10+ с [python.org](https://www.python.org/downloads/)
- При установке **обязательно отметьте** "Add Python to PATH"
- Проверьте установку: `python --version`

#### 2. Установка CUDA (если есть NVIDIA GPU)
- Скачайте CUDA Toolkit 11.8+ с [nvidia.com](https://developer.nvidia.com/cuda-downloads)
- Установите драйверы NVIDIA
- Проверьте установку: `nvidia-smi`

#### 3. Установка зависимостей
```bash
# В папке проекта
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### 4. Настройка API ключа
```bash
set OPENAI_API_KEY=your_api_key_here
```

## 🚀 Запуск системы

### Быстрый запуск (рекомендуется)
```bash
python quick_start.py
```

### Ручной запуск отдельных компонентов

#### 1. Только анализ документов (без обучения)
```bash
python run_large_pipeline.py --skip-training
```

#### 2. Полный пайплайн (анализ + обучение)
```bash
python run_large_pipeline.py --epochs 15 --batch-size 1
```

#### 3. Запуск GUI
```bash
python gui/app.py
```

#### 4. Мониторинг обучения
```bash
python src/monitor_training.py --open_browser
```

#### 5. Демо-режим
```bash
python demo.py
```

## 📊 Оптимальные настройки для больших данных

### Для мощного компьютера (RTX 4090, 64GB RAM):
```bash
python run_large_pipeline.py \
  --max-docs 1000 \
  --max-workers 5 \
  --epochs 20 \
  --batch-size 2
```

### Для среднего компьютера (RTX 3080, 32GB RAM):
```bash
python run_large_pipeline.py \
  --max-docs 500 \
  --max-workers 3 \
  --epochs 15 \
  --batch-size 1
```

### Для слабого компьютера (CPU, 16GB RAM):
```bash
python run_large_pipeline.py \
  --max-docs 100 \
  --max-workers 1 \
  --epochs 10 \
  --batch-size 1
```

## 🔧 Настройка производительности

### Оптимизация для GPU
В файле `config/optimal_settings.json` можно настроить:
- `use_flash_attention`: true (ускоряет внимание)
- `use_8bit_optimizer`: true (экономит память)
- `use_gradient_checkpointing`: true (экономит память)

### Оптимизация для CPU
```json
{
  "optimization_config": {
    "use_flash_attention": false,
    "use_8bit_optimizer": false,
    "use_4bit_quantization": true,
    "max_memory_usage": "8GB"
  }
}
```

## 📁 Структура проекта

```
LLM-lawyer/
├── data/                    # Данные
│   ├── raw/                # Исходные PDF файлы
│   ├── processed/          # Обработанные тексты
│   ├── analyzed/           # Результаты анализа OpenAI
│   └── structured/         # Структурированные данные
├── models/                 # Обученные модели
├── results/                # Результаты тестирования
├── src/                    # Исходный код
├── gui/                    # Графический интерфейс
├── config/                 # Конфигурационные файлы
├── quick_start.py          # Быстрый запуск
├── run_large_pipeline.py   # Основной пайплайн
└── requirements.txt        # Зависимости
```

## 🐛 Решение проблем

### Проблема: "CUDA out of memory"
**Решение:**
```bash
# Уменьшите размер батча
python run_large_pipeline.py --batch-size 1

# Или используйте CPU
export CUDA_VISIBLE_DEVICES=""
```

### Проблема: "OpenAI API key not found"
**Решение:**
```bash
# Установите переменную окружения
set OPENAI_API_KEY=your_key_here

# Или создайте файл .env
echo OPENAI_API_KEY=your_key_here > .env
```

### Проблема: "Module not found"
**Решение:**
```bash
# Переустановите зависимости
pip install -r requirements.txt --force-reinstall
```

### Проблема: "Permission denied"
**Решение:**
```bash
# Запустите от имени администратора
# Или измените права доступа к папке
```

## 📞 Поддержка

При возникновении проблем:

1. **Проверьте логи** в консоли
2. **Убедитесь в корректности** API ключа
3. **Проверьте свободное место** на диске
4. **Убедитесь в стабильности** интернет-соединения
5. **Обратитесь к документации** в папке проекта

## 🎯 Рекомендации по использованию

### Для максимального качества:
1. **Используйте качественные данные** - хорошо структурированные PDF документы
2. **Обрабатывайте большие объемы** - минимум 500-1000 документов
3. **Настройте параметры** под ваши данные в `config/optimal_settings.json`
4. **Мониторьте обучение** через TensorBoard
5. **Тестируйте модель** на различных примерах

### Для экономии ресурсов:
1. **Начните с малого** - обработайте 50-100 документов
2. **Используйте CPU** если GPU недоступен
3. **Уменьшите размер модели** в конфигурации
4. **Используйте квантизацию** для экономии памяти

## ✅ Проверка готовности системы

Запустите быстрый тест:
```bash
python quick_test.py
```

Если тест прошел успешно, система готова к работе!

---

**🎉 Система готова к использованию!**

Для начала работы запустите:
```bash
python quick_start.py
``` 