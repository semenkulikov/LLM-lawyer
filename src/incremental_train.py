#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import numpy as np
from loguru import logger

def setup_logging():
    """Настройка логирования"""
    logger.remove()
    logger.add(
        "logs/incremental_training.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(lambda msg: print(msg, end=""), level="INFO")

def load_existing_model(model_path):
    """
    Загрузка существующей модели для дообучения
    
    Args:
        model_path: Путь к существующей модели
        
    Returns:
        model: Загруженная модель
        tokenizer: Загруженный токенизатор
    """
    logger.info(f"Загрузка существующей модели из {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Определяем тип модели по конфигурации
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        
        if config.model_type in ['gpt2', 'gpt']:
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(model_path)
        elif config.model_type in ['bart', 't5', 'pegasus', 'marian']:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            # Пробуем загрузить как обычную модель для генерации
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Проверяем, есть ли pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Модель успешно загружена (тип: {config.model_type})")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

def load_new_data(data_file):
    """
    Загрузка новых данных для дообучения
    
    Args:
        data_file: Путь к файлу с новыми данными (JSONL формат)
        
    Returns:
        list: Список примеров для обучения
    """
    logger.info(f"Загрузка новых данных из {data_file}")
    
    examples = []
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    if 'facts' in example and 'reasoning' in example:
                        examples.append(example)
                    else:
                        logger.warning(f"Пропущена строка {line_num}: отсутствуют обязательные поля")
                except json.JSONDecodeError:
                    logger.warning(f"Пропущена строка {line_num}: некорректный JSON")
                    
        logger.info(f"Загружено {len(examples)} новых примеров")
        return examples
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        raise

def prepare_dataset(examples, tokenizer, max_input_length=1024, max_output_length=1024):
    """
    Подготовка датасета для обучения
    
    Args:
        examples: Список примеров
        tokenizer: Токенизатор
        max_input_length: Максимальная длина входной последовательности
        max_output_length: Максимальная длина выходной последовательности
        
    Returns:
        Dataset: Подготовленный датасет
    """
    logger.info("Подготовка датасета для обучения")
    
    def tokenize_function(examples):
        # Для GPT2 и подобных моделей объединяем вход и выход
        combined_texts = []
        for fact, reasoning in zip(examples["facts"], examples["reasoning"]):
            # Формат: "Факты: {facts}\nМотивировка: {reasoning}"
            combined_text = f"Факты: {fact}\nМотивировка: {reasoning}"
            combined_texts.append(combined_text)
    
        # Токенизация объединенного текста
        tokenized = tokenizer(
            combined_texts,
            max_length=max_input_length + max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Для GPT2 labels = input_ids
        labels = tokenized["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
    
    # Создаем датасет
    dataset = Dataset.from_list(examples)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Датасет подготовлен: {len(tokenized_dataset)} примеров")
    return tokenized_dataset

def create_backup(model_path, backup_dir="models/backups"):
    """
    Создание резервной копии модели перед дообучением
    
    Args:
        model_path: Путь к модели
        backup_dir: Директория для резервных копий
        
    Returns:
        str: Путь к резервной копии
    """
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}"
    backup_full_path = backup_path / backup_name
    
    logger.info(f"Создание резервной копии: {backup_full_path}")
    
    # Копируем модель
    import shutil
    shutil.copytree(model_path, backup_full_path)
    
    logger.info(f"Резервная копия создана: {backup_full_path}")
    return str(backup_full_path)

def incremental_train(model, tokenizer, train_dataset, output_dir, 
                     epochs=3, batch_size=4, learning_rate=5e-5,
                     warmup_steps=100, save_steps=500, eval_steps=500):
    """
    Инкрементальное дообучение модели
    
    Args:
        model: Модель для дообучения
        tokenizer: Токенизатор
        train_dataset: Датасет для обучения
        output_dir: Директория для сохранения результатов
        epochs: Количество эпох
        batch_size: Размер батча
        learning_rate: Скорость обучения
        warmup_steps: Количество шагов для разогрева
        save_steps: Частота сохранения модели
        eval_steps: Частота оценки модели
    """
    logger.info("Начало инкрементального дообучения")
    
    # Создаем директорию для результатов
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Настройки обучения
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=100,
        save_total_limit=3,  # Сохраняем только 3 лучшие модели
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # Отключаем wandb для инкрементального обучения
    )
    
    # Data collator (адаптируем под тип модели)
    from transformers import DataCollatorForLanguageModeling
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Не используем masked language modeling для GPT2
    )
    
    # Создаем trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Запускаем обучение
    logger.info("Запуск обучения...")
    train_result = trainer.train()
    
    # Сохраняем модель
    trainer.save_model()
    tokenizer.save_pretrained(output_path)
    
    # Логируем результаты
    logger.info("Обучение завершено")
    logger.info(f"Общее время обучения: {train_result.metrics['train_runtime']:.2f} секунд")
    logger.info(f"Средняя скорость обучения: {train_result.metrics['train_runtime']/train_result.metrics['train_steps']:.4f} секунд/шаг")
    
    return train_result

def main():
    parser = argparse.ArgumentParser(description='Инкрементальное дообучение модели на новых данных')
    
    # Обязательные аргументы
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Путь к существующей модели для дообучения')
    parser.add_argument('--new_data', type=str, required=True, 
                       help='Путь к файлу с новыми данными (JSONL формат)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Директория для сохранения дообученной модели')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох (оптимизировано для больших датасетов)')
    parser.add_argument('--batch_size', type=int, default=8, help='Размер батча (оптимизировано для больших датасетов)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Скорость обучения (оптимизировано для больших датасетов)')
    parser.add_argument('--max_input_length', type=int, default=1024, help='Максимальная длина входной последовательности')
    parser.add_argument('--max_output_length', type=int, default=1024, help='Максимальная длина выходной последовательности')
    
    # Дополнительные параметры
    parser.add_argument('--warmup_steps', type=int, default=100, help='Количество шагов для разогрева')
    parser.add_argument('--save_steps', type=int, default=500, help='Частота сохранения модели')
    parser.add_argument('--eval_steps', type=int, default=500, help='Частота оценки модели')
    parser.add_argument('--create_backup', action='store_true', help='Создать резервную копию модели')
    parser.add_argument('--backup_dir', type=str, default='models/backups', help='Директория для резервных копий')
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging()
    
    try:
        # Создание резервной копии, если указан флаг
        if args.create_backup:
            backup_path = create_backup(args.model_path, args.backup_dir)
        
        # Загрузка существующей модели
        model, tokenizer = load_existing_model(args.model_path)
        
        # Загрузка новых данных
        new_examples = load_new_data(args.new_data)
        
        if not new_examples:
            logger.error("Нет данных для дообучения")
        return
    
        # Подготовка датасета
        train_dataset = prepare_dataset(
            new_examples, 
            tokenizer, 
            args.max_input_length, 
            args.max_output_length
        )
    
        # Инкрементальное обучение
        train_result = incremental_train(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps
        )
    
        logger.info(f"Инкрементальное дообучение завершено успешно!")
        logger.info(f"Дообученная модель сохранена в: {args.output_dir}")
        
        # Сохраняем информацию о дообучении
        training_info = {
            "original_model": args.model_path,
            "new_data_file": args.new_data,
            "training_date": datetime.now().isoformat(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "new_examples_count": len(new_examples),
            "training_metrics": train_result.metrics
        }
        
        if args.create_backup:
            training_info["backup_path"] = backup_path
        
        info_file = Path(args.output_dir) / "incremental_training_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Информация о дообучении сохранена в: {info_file}")
        
    except Exception as e:
        logger.error(f"Ошибка при инкрементальном дообучении: {str(e)}")
        raise

if __name__ == '__main__':
    main() 