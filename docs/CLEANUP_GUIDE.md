# 🧹 Руководство по очистке данных

## Быстрые команды

### Показать статус директорий
```bash
python clean_data.py --status
```

### Очистить структурированные данные (чаще всего используется)
```bash
python clean_data.py --processed --force
```

### Очистить модель
```bash
python clean_data.py --model --force
```

### Очистить результаты тестирования
```bash
python clean_data.py --results --force
```

### Очистить все промежуточные данные
```bash
python clean_data.py --all --force
```

## Что очищается

### `--processed` (структурированные данные)
- `data/structured/` - структурированные данные
- `data/analyzed/` - проанализированные JSON файлы
- `data/train_dataset.jsonl` - обучающий датасет
- `data/train_dataset_test.jsonl` - тестовый датасет
- `data/train_dataset_meta.json` - метаданные датасета

### `--model` (данные модели)
- `models/legal_model/` - обученная модель
- `models/legal_model/logs/` - логи обучения

### `--results` (результаты)
- `results/` - результаты тестирования

### `--all` (все промежуточные данные)
- Все вышеперечисленное

## ⚠️ Важно

- **Исходные PDF файлы в `data/raw/` НЕ удаляются**
- **Обработанные текстовые файлы в `data/processed/` НЕ удаляются**
- Используйте `--force` для автоматического подтверждения
- Без `--force` будет запрошено подтверждение

## Типичный сценарий использования

1. Добавили новые PDF в `data/raw/`
2. Очистили старые структурированные данные:
   ```bash
   python clean_data.py --processed --force
   ```
3. Запустили пайплайн заново:
   ```bash
   python run_pipeline.py --max-docs 3 --epochs 3
   ``` 