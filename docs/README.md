# 🤖 Юридический ассистент - Универсальная гибридная система

Современная система для генерации юридических документов с использованием гибридного подхода: локальная модель QVikhr + внешний LLM (OpenAI/Gemini).

## 📁 Структура проекта

```
LLM-lawyer/
├── 📁 src/                    # Основной код
│   ├── inference.py           # Инференс модели
│   ├── hybrid_processor.py    # Гибридный процессор
│   ├── train.py              # Обучение модели
│   └── ...
├── 📁 gui/                    # Графический интерфейс
│   └── legal_assistant_gui.py # Единый GUI файл
├── 📁 scripts/                # Batch файлы
│   ├── start_gui.bat         # Запуск GUI
│   ├── run_pipeline.bat      # Запуск пайплайна
│   └── ...
├── 📁 tests/                  # Тесты
│   ├── test_hybrid_universal.py
│   ├── test_quick_connection.py
│   └── ...
├── 📁 docs/                   # Документация
│   ├── README.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── ...
├── 📁 data/                   # Данные
├── 📁 models/                 # Обученные модели
├── 📁 results/                # Результаты
└── 📁 gui_old/                # Старые версии GUI
```

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка API ключей
Создайте файл `.env` на основе `env.example`:
```bash
cp env.example .env
```

Добавьте ваши API ключи:
```env
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### 3. Запуск GUI
```bash
scripts/start_gui.bat
```

Или напрямую:
```bash
python gui/legal_assistant_gui.py
```

## 🎨 Особенности GUI

### ✅ Исправленные проблемы:
- **Убраны эмодзи** - чистый интерфейс без квадратных скобок
- **Поддержка Markdown** - красивое форматирование ответов
- **Улучшенный дизайн** - современные стили и анимации
- **Реальный прогресс** - отображение этапов обработки
- **Красивые элементы** - улучшенные выпадающие списки и галочки

### 🎯 Функциональность:
- **Гибридная обработка** - локальная модель + внешний LLM
- **Выбор провайдера** - OpenAI или Gemini
- **Режимы обработки** - полировка, расширение, проверка
- **Поддержка Markdown** - заголовки, жирный текст, курсив, списки
- **Сохранение результатов** - в текстовом или HTML формате

## 🔧 Технические особенности

### Гибридный подход:
1. **Локальная модель** (QVikhr-3-4B) - базовая генерация
2. **Внешний LLM** (OpenAI/Gemini) - улучшение и полировка
3. **Универсальный процессор** - автоматический выбор провайдера

### Поддерживаемые форматы:
- **Ввод**: текстовые файлы (.txt)
- **Вывод**: текст (.txt) или HTML (.html) с форматированием

### Режимы обработки:
- **Полировка** - исправление ошибок и улучшение стиля
- **Расширение** - добавление недостающих элементов
- **Проверка** - проверка на ошибки с комментариями

## 📚 Документация

- `docs/DEPLOYMENT_GUIDE.md` - Руководство по развертыванию
- `docs/INSTALLATION_GUIDE.md` - Подробная установка
- `docs/CUSTOMER_GUIDE.md` - Руководство пользователя

## 🧪 Тестирование

```bash
# Тест гибридной системы
python tests/test_hybrid_universal.py

# Быстрая проверка подключения
python tests/test_quick_connection.py
```

### Для заказчика:
1. **Скачайте проект** и распакуйте в удобное место
2. **Откройте командную строку** в папке проекта
3. **Запустите автоматическую установку**:
   ```cmd
   setup_project.bat
   ```
4. **Настройте API ключи** в файле `.env`
5. **Поместите PDF документы** в папку `data\raw`
6. **Запустите обучение**:
   ```cmd
   run_pipeline.bat
   ```
7. **Используйте модель**:
   ```cmd
   start_gui.bat
   ```

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
python src\train.py --train_file data\train_dataset.jsonl --test_file data\train_dataset_test.jsonl --output_dir models\legal_model --epochs 3 
--batch_size 1 --gradient_accumulation_steps 8 --max_length 2048

# Распределенное обучение на нескольких GPU
python run_distributed_training.py

# Генерация мотивировочной части (оптимизированная)
python src\inference.py --model_path models\legal_model --input_file input.txt --output_file output.txt --temperature 0.3 --num_beams 8
# Тестирование модели
python src\test_example.py --model_path models\legal_model --test_file data\train_dataset_test.jsonl --output_dir results
# Мониторинг обучения
python src\monitor_training.py --log_dir models --port 6006 --open_browser
# Мониторинг прогресса обработки
python monitor_progress.py
# Инкрементальное дообучение модели
python src\incremental_train.py --model_path models\legal_model --new_data data\new_data.jsonl --output_dir models\incremental --epochs 3
# Автоматическое дообучение (мониторинг файлов)
python src\auto_incremental.py --watch
# Автоматическое дообучение (планировщик)
python src\auto_incremental.py --schedule


## 🔄 Обновления

### v2.0 - Унификация проекта
- ✅ Реорганизована структура проекта
- ✅ Единый GUI файл вместо множественных версий
- ✅ Исправлены проблемы с отображением
- ✅ Добавлена полная поддержка Markdown
- ✅ Улучшен дизайн и UX

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи в консоли
2. Убедитесь в корректности API ключей
3. Проверьте наличие модели в `models/legal_model/`

---

**Разработано для юридических профессионалов** ⚖️ 