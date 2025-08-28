# Быстрый старт с QVikhr-3-4B

## Требования

- **GPU с 24GB+ памяти** (RTX 4090, A100, H100)
- **PyTorch 2.7+** с CUDA поддержкой
- **Python 3.8+**

## Установка

```bash
# Активируйте виртуальное окружение
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Установите зависимости
pip install -r requirements.txt
```

## Обучение модели

### На одном GPU (24GB+ памяти)

```bash
python src/train.py \
  --train_file data/train_dataset.jsonl \
  --test_file data/train_dataset_test.jsonl \
  --output_dir models/legal_model \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 2048 \
  --learning_rate 1e-5
```

### На нескольких GPU (распределенное обучение)

```bash
python run_distributed_training.py
```

## Генерация текста

```bash
python src/inference.py \
  --model_path models/legal_model \
  --input_text "Истец требует взыскать долг 100000 рублей" \
  --max_output_length 1024
```

## Параметры для больших объемов данных

### Для 500,000+ документов:

```bash
# Увеличьте количество эпох
--epochs 10

# Увеличьте длину последовательности
--max_length 4096

# Уменьшите learning rate
--learning_rate 5e-6

# Увеличьте gradient accumulation
--gradient_accumulation_steps 16
```

### Мониторинг обучения:

```bash
python src/monitor_training.py --log_dir models --port 6006 --open_browser
```

## Ожидаемые результаты

После обучения на 500,000+ документах модель должна генерировать:

- **Связанный и логичный текст** на русском языке
- **Правильную юридическую терминологию**
- **Структурированную мотивировочную часть**
- **Ссылки на соответствующие статьи закона**

## Устранение проблем

### Нехватка памяти GPU:
- Уменьшите `batch_size` до 1
- Увеличьте `gradient_accumulation_steps`
- Уменьшите `max_length`

### Медленное обучение:
- Используйте `fp16=True` (уже включено)
- Увеличьте количество GPU
- Используйте распределенное обучение

### Плохое качество генерации:
- Увеличьте количество эпох
- Увеличьте размер датасета
- Настройте параметры генерации (temperature, top_p) 