# Юридический ассистент - Генератор мотивировки

Система для автоматической генерации мотивировочной части судебных решений на основе фактических обстоятельств дела с использованием модели QVikhr (дообученной на юридических данных).

## 🚀 Поддерживаемые модели

### T5 (базовая)
- **Модель**: `t5-small`
- **Размер**: ~60MB
- **Качество**: Базовое, подходит для тестирования
- **Требования**: Минимальные

### QVikhr (продвинутая) ⭐
- **Модель**: `Vikhrmodels/QVikhr-3-4B-Instruction`
- **Размер**: ~8GB
- **Качество**: Высокое, специализирована на русском языке
- **Особенности**: 
  - 4.02B параметров
  - Инструктивный тюнинг
  - Оценка 78.2 в Ru Arena General
  - Оптимизирована для юридических задач
- **Требования**: GPU с 8GB+ памяти

## Описание проекта

Проект представляет собой полный пайплайн обработки юридических документов:

1. **Извлечение текста** из PDF документов
2. **Анализ структуры** с помощью OpenAI API
3. **Создание датасета** для обучения
4. **Обучение модели** на парах "факты → мотивировка"
5. **Генерация мотивировки** для новых случаев

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка OpenAI API
```bash
# Windows
set OPENAI_API_KEY=sk-your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Подготовка данных
Поместите PDF документы в папку `data/raw/`

### 4. Запуск полного пайплайна
```bash
python run_pipeline.py --max-docs 10 --epochs 5
```

### 5. Запуск GUI
```bash
python gui/app.py
```

## 📖 Подробные инструкции

См. файл [GETTING_STARTED.md](GETTING_STARTED.md) для подробных инструкций по использованию.

## 🔧 Диагностика

```bash
# Проверка системы
python quick_test.py

# Проверка CUDA
python check_cuda.py

# Тестирование OpenAI API
python src/test_openai_key.py
```

## 🚀 Быстрый запуск (рекомендуется)

### Для заказчика:
1. **Скачайте проект** и распакуйте в удобное место
2. **Откройте командную строку** в папке проекта
3. **Запустите автоматическую установку**:
   ```bash
   python quick_start.py
   ```
4. **Следуйте инструкциям** на экране

### Для разработчика:

#### Полный пайплайн (оптимизированный):
```bash
# Автоматический запуск всего пайплайна
python run_pipeline.py --epochs 15 --batch-size 1

# Только анализ документов (без обучения)
python run_pipeline.py --skip-training

# Настройка для больших данных
python run_pipeline.py --max-docs 1000 --max-workers 5 --epochs 20
```

#### Отдельные компоненты:
```bash
# Предобработка PDF документов
python src\preprocess.py --input-dir data\raw --output-dir data\processed

# Параллельный анализ с OpenAI
python src\process_large_dataset.py --input-dir data\processed --output-dir data\analyzed --max-workers 3

# Создание обучающего датасета
python src\build_dataset.py --analyzed-dir data\analyzed --output-file data\train_dataset.jsonl

# Обучение модели QVikhr-3-4B (один GPU)
python src\train.py --train_file data\train_dataset.jsonl --test_file data\train_dataset_test.jsonl --output_dir models\legal_model --epochs 3 --batch_size 1 --gradient_accumulation_steps 8 --max_length 2048

# Распределенное обучение на нескольких GPU
python run_distributed_training.py

# Генерация мотивировочной части (оптимизированная)
python src\inference.py --model_path models\legal_model --input_file input.txt --output_file output.txt --temperature 0.3 --num_beams 8

# Тестирование модели
python src\test_example.py --model_path models\legal_model --test_file data\train_dataset_test.jsonl --output_dir results

# Мониторинг обучения
python src\monitor_training.py --log_dir models --port 6006 --open_browser

# Инкрементальное дообучение модели
python src\incremental_train.py --model_path models\legal_model --new_data data\new_data.jsonl --output_dir models\incremental --epochs 3

# Автоматическое дообучение (мониторинг файлов)
python src\auto_incremental.py --watch

# Автоматическое дообучение (планировщик)
python src\auto_incremental.py --schedule

# Управление версиями моделей
python src\model_manager.py list
python src\model_manager.py backup models\legal_model
python src\model_manager.py cleanup --keep 5

# Запуск GUI
python gui\app.py
```

#### Быстрый тест системы:
```bash
python quick_test.py
```

#### Интерактивное меню (если нужно):
```bash
# Можно добавить в run_pipeline.py функцию show_menu() если потребуется
```

## Использование обученной модели

### Через командную строку

```bash
# Использует QVikhr по умолчанию
python src/inference.py --input_file input.txt --output_file output.txt
```

### Через GUI

```bash
python gui/app.py
```

## Структура проекта

```
LLM-lawyer/
├── src/                    # Исходный код
│   ├── preprocess.py      # Извлечение текста из PDF
│   ├── process_with_openai.py  # Анализ документов
│   ├── build_dataset.py   # Создание датасета
│   ├── train.py          # Обучение модели
│   ├── inference.py      # Генерация мотивировки
│   ├── test_example.py   # Тестирование модели
│   └── monitor_training.py # Мониторинг обучения
├── gui/                   # Графический интерфейс
│   └── app.py            # Основное приложение
├── data/                  # Данные
│   ├── raw/              # Исходные PDF файлы
│   ├── processed/        # Извлеченный текст
│   ├── analyzed/         # Результаты анализа OpenAI
│   └── structured/       # Структурированные данные
├── models/               # Обученные модели
├── results/              # Результаты тестирования
├── run_pipeline.py       # Основной скрипт пайплайна
├── requirements.txt      # Зависимости
└── README.md            # Документация
```

## Требования

### Системные требования
- Python 3.8+
- 8GB RAM (рекомендуется)
- GPU для ускорения обучения (опционально)

### Основные зависимости
- `torch` - PyTorch для нейронных сетей
- `transformers` - Hugging Face Transformers
- `openai` - OpenAI API клиент
- `pymupdf` - Работа с PDF файлами
- `nltk` - Обработка естественного языка
- `tqdm` - Прогресс-бары
- `numpy` - Численные вычисления

## Инкрементальное дообучение

Система поддерживает автоматическое дообучение модели на новых данных:

### Ручное дообучение
```bash
python src\incremental_train.py --model_path models\legal_model --new_data data\new_data.jsonl --output_dir models\incremental --epochs 3
```

### Автоматическое дообучение
- **Мониторинг файлов**: Автоматически обрабатывает новые файлы данных
- **Планировщик**: Запускает дообучение по расписанию
- **Резервные копии**: Автоматическое создание бэкапов перед дообучением

### Управление версиями моделей
```bash
# Список всех моделей
python src\model_manager.py list

# Создание резервной копии
python src\model_manager.py backup models\legal_model

# Очистка старых версий
python src\model_manager.py cleanup --keep 5

# Сравнение моделей
python src\model_manager.py compare models\model_v1 models\model_v2
```

## Мониторинг обучения

Для отслеживания процесса обучения:

```bash
python src/monitor_training.py --log_dir models --port 6006 --open_browser
```

Затем откройте http://localhost:6006 в браузере.

## Примеры использования

### Входные данные (факты)
```
Истец обратился в суд с требованием о взыскании задолженности по договору займа в размере 100 000 рублей. 
Договор был заключен 15.01.2023, срок возврата - 15.04.2023. 
Ответчик не возвратил заем в установленный срок.
```

### Выходные данные (мотивировка)
```
Суд считает требования истца обоснованными и подлежащими удовлетворению.

В соответствии со статьей 807 ГК РФ по договору займа одна сторона (займодавец) передает в собственность другой стороне (заемщику) деньги или другие вещи, определенные родовыми признаками, а заемщик обязуется возвратить займодавцу такую же сумму денег (сумму займа) или равное количество полученных им вещей того же рода и качества.

Согласно статье 810 ГК РФ заемщик обязан возвратить займодавцу полученную сумму займа в срок и в порядке, которые предусмотрены договором займа.

Материалами дела подтверждается, что между сторонами был заключен договор займа, ответчик получил денежные средства, но не возвратил их в установленный срок.

При таких обстоятельствах суд приходит к выводу о необходимости удовлетворить исковые требования в полном объеме.
```

## Ограничения и особенности

1. **Качество анализа** зависит от качества исходных PDF документов
2. **Стоимость OpenAI API** - используйте ограничение `--max-docs` для контроля расходов
3. **Время обучения** - зависит от размера датасета и количества эпох
4. **Требования к данным** - документы должны быть на русском языке

## Устранение неполадок

### Быстрая диагностика
```bash
# Запустите полную диагностику системы
python quick_test.py
```

### Ошибка "Модель не найдена"
```bash
# Убедитесь, что модель обучена
ls models/legal_model/
```

### Ошибка OpenAI API
```bash
# Проверьте API ключ
python src/test_openai_key.py
```

### Ошибки кодировки
```bash
# Система автоматически обрабатывает проблемы с кодировкой
# Если возникают ошибки Unicode, проверьте исходные файлы
```

### Недостаточно памяти
```bash
# Уменьшите размер батча
python src/train.py --batch_size 2
```

## Лицензия

MIT License - см. файл LICENSE для подробностей.

## Поддержка

При возникновении проблем:
1. Проверьте логи выполнения
2. Убедитесь в корректности входных данных
3. Проверьте наличие всех зависимостей
4. Убедитесь в валидности OpenAI API ключа