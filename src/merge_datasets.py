#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
from pathlib import Path
from loguru import logger

def merge_datasets(input_dir="datasets/hybrid_generated", output_file="datasets/merged_training_dataset.jsonl"):
    """
    Объединение всех датасетов в один файл для обучения
    
    Args:
        input_dir: Папка с датасетами
        output_file: Выходной файл
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Папка {input_dir} не существует")
        return False
    
    # Создаем папку для выходного файла
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ищем все JSONL файлы
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        logger.warning(f"В папке {input_dir} не найдено JSONL файлов")
        return False
    
    logger.info(f"Найдено {len(jsonl_files)} файлов датасета")
    
    total_examples = 0
    total_input_chars = 0
    total_output_chars = 0
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for jsonl_file in jsonl_files:
            logger.info(f"Обрабатываю файл: {jsonl_file.name}")
            
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as infile:
                    for line_num, line in enumerate(infile, 1):
                        if line.strip():
                            try:
                                record = json.loads(line)
                                
                                # Проверяем структуру записи
                                if 'instruction' in record and 'input' in record and 'output' in record:
                                    # Записываем в объединенный файл
                                    outfile.write(line)
                                    total_examples += 1
                                    total_input_chars += len(record.get('input', ''))
                                    total_output_chars += len(record.get('output', ''))
                                else:
                                    logger.warning(f"Пропущена некорректная запись в {jsonl_file.name}:{line_num}")
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"Ошибка JSON в {jsonl_file.name}:{line_num}: {e}")
                                continue
                                
            except Exception as e:
                logger.error(f"Ошибка чтения файла {jsonl_file}: {e}")
                continue
    
    # Статистика
    avg_input_length = total_input_chars // total_examples if total_examples > 0 else 0
    avg_output_length = total_output_chars // total_examples if total_examples > 0 else 0
    
    logger.info(f"""
📊 Результат объединения датасетов:

📁 Выходной файл: {output_path}
📝 Всего примеров: {total_examples}
📏 Средняя длина ввода: {avg_input_length} символов
📏 Средняя длина вывода: {avg_output_length} символов
📊 Общий объем ввода: {total_input_chars} символов
📊 Общий объем вывода: {total_output_chars} символов

✅ Датасет готов для обучения!
    """)
    
    return True

def create_training_config(output_file="datasets/training_config.json"):
    """Создание конфигурации для обучения"""
    config = {
        "dataset_path": "datasets/merged_training_dataset.jsonl",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "training_args": {
            "output_dir": "models/legal_model_v2",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 2,
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_pin_memory": False
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    }
    
    config_path = Path(output_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Конфигурация для обучения сохранена в: {config_path}")
    return config_path

if __name__ == "__main__":
    # Объединяем датасеты
    success = merge_datasets()
    
    if success:
        # Создаем конфигурацию для обучения
        create_training_config()
        
        print("\n🎉 Готово! Теперь можно обучать модель:")
        print("python src/train.py --config datasets/training_config.json")
    else:
        print("\n❌ Ошибка при объединении датасетов")
