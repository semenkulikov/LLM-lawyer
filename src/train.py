#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
from loguru import logger
import shutil
from datetime import datetime

def process_dataset_for_hf_trainer(dataset_path, tokenizer, max_length=2048):
    """Обработка датасета для HF Trainer с увеличенной длиной последовательности"""
    data = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line)
                if 'facts' in example and 'reasoning' in example:
                    data.append({
                        'facts': example['facts'],
                        'reasoning': example['reasoning']
                    })
            except json.JSONDecodeError:
                continue
    
    hf_dataset = HFDataset.from_list(data)
    
    def preprocess_function(examples):
        texts = []
        for facts, reasoning in zip(examples['facts'], examples['reasoning']):
            text = f"Факты: {facts}\nМотивировка: {reasoning}"
            texts.append(text)
        
        # Безопасная токенизация с проверкой длины
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Проверяем что все токены в допустимых пределах
        if tokenized["input_ids"].max() >= tokenizer.vocab_size:
            logger.warning(f"Обнаружены токены вне словаря: {tokenized['input_ids'].max()}")
            # Обрезаем до размера словаря
            tokenized["input_ids"] = torch.clamp(tokenized["input_ids"], 0, tokenizer.vocab_size - 1)
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    processed_dataset = hf_dataset.map(
        preprocess_function,
        batched=True,
        desc="Обработка датасета",
        remove_columns=hf_dataset.column_names
    )
    
    return processed_dataset

def backup_previous_model(output_dir):
    """Создание резервной копии предыдущей модели"""
    if os.path.exists(output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{output_dir}_backup_{timestamp}"
        logger.info(f"Создание резервной копии: {backup_dir}")
        shutil.copytree(output_dir, backup_dir)
        return backup_dir
    return None

def load_existing_model(model_name, output_dir, tokenizer):
    """Загрузка существующей модели для дообучения"""
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        logger.info(f"Загрузка существующей модели из {output_dir}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                output_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            logger.info("Существующая модель загружена успешно")
            return model
        except Exception as e:
            logger.warning(f"Не удалось загрузить существующую модель: {e}")
            logger.info("Загружаем базовую модель для обучения с нуля")
    
    # Загружаем базовую модель
    logger.info(f"Загрузка базовой модели {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

def train_model(args):
    """Обучение модели QVikhr-3-4B с оптимизациями для RTX 30 Series"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"PyTorch версия: {torch.__version__}")
    logger.info(f"CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA версия: {torch.version.cuda}")
        logger.info(f"Количество GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i} память: {gpu_memory:.1f} GB")
            
            # Проверяем достаточность памяти для RTX 30 Series
            if gpu_memory < 8:
                logger.warning(f"GPU {i} имеет мало памяти ({gpu_memory:.1f} GB). Рекомендуется 8GB+ для стабильной работы")
            else:
                logger.info(f"✅ GPU {i} имеет достаточно памяти ({gpu_memory:.1f} GB) для QVikhr-3-4B")
    else:
        logger.warning("CUDA недоступна! Обучение будет происходить на CPU (медленнее)")
        logger.info("Рекомендуется установить PyTorch с поддержкой CUDA для ускорения")
    
    # Загружаем токенизатор
    logger.info("Загрузка токенизатора QVikhr...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Создаем резервную копию предыдущей модели
    if args.resume_training:
        backup_previous_model(args.output_dir)
    
    # Загружаем модель (существующую или базовую)
    model = load_existing_model(args.model_name, args.output_dir, tokenizer)
    
    # Включаем gradient checkpointing для экономии памяти
    model.gradient_checkpointing_enable()
    
    logger.info("Модель загружена с автоматическим распределением памяти")
    
    # Обрабатываем датасеты
    train_dataset = process_dataset_for_hf_trainer(args.train_file, tokenizer, args.max_length)
    val_dataset = process_dataset_for_hf_trainer(args.test_file, tokenizer, args.max_length)
    
    # Оптимизированные настройки для RTX 30 Series ноутбука
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=5,  # Уменьшено для более частого логирования
        save_steps=50,    # Уменьшено для более частого сохранения
        eval_steps=25,    # Уменьшено для более частой валидации
        save_total_limit=5,  # Сохраняем больше чекпоинтов для дообучения
        seed=args.seed,
        
        # Mixed precision training
        fp16=True,
        bf16=False,
        
        # Gradient accumulation для эффективного использования памяти
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Gradient checkpointing
        gradient_checkpointing=True,
        
        # Оптимизации памяти для ноутбука
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        
        # Консервативные параметры обучения
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        
        # Отчеты
        report_to=[],  # Отключаем wandb для экономии ресурсов
        logging_dir=f"{args.output_dir}/logs",
        
        # Сохранение
        save_strategy="steps",
        eval_strategy="steps",
        
        # Метрики
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Дополнительные оптимизации для ноутбука
        dataloader_num_workers=0,  # Отключаем multiprocessing для стабильности
        group_by_length=True,      # Группировка по длине для эффективности
    )
    
    # Создаем тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Обучаем
    logger.info("Начало обучения QVikhr-3-4B...")
    trainer.train()
    
    # Сохраняем
    logger.info(f"Сохранение модели в {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Сохраняем информацию о дообучении
    training_info = {
        "last_training_date": datetime.now().isoformat(),
        "epochs_completed": args.epochs,
        "total_steps": trainer.state.global_step,
        "final_loss": trainer.state.log_history[-1]["train_loss"] if trainer.state.log_history else None,
        "model_path": args.output_dir,
        "resume_training": args.resume_training
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w", encoding="utf-8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    
    logger.info("Обучение завершено!")
    logger.info(f"Модель сохранена в: {args.output_dir}")
    logger.info(f"Информация о дообучении: {training_info}")

def main():
    parser = argparse.ArgumentParser(description='Обучение QVikhr-3-4B для генерации мотивировочной части (оптимизировано для RTX 30 Series)')
    
    parser.add_argument('--train_file', type=str, required=True, help='Путь к обучающему файлу')
    parser.add_argument('--test_file', type=str, required=True, help='Путь к тестовому файлу')
    parser.add_argument('--output_dir', type=str, required=True, help='Директория для сохранения')
    parser.add_argument('--model_name', type=str, default='Vikhrmodels/QVikhr-3-4B-Instruction', help='QVikhr-3-4B модель')
    parser.add_argument('--max_length', type=int, default=2048, help='Максимальная длина последовательности')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох (оптимизировано для больших датасетов)')
    parser.add_argument('--batch_size', type=int, default=8, help='Размер батча (оптимизировано для больших датасетов)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Шаги накопления градиента (оптимизировано для больших датасетов)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Скорость обучения (оптимизировано для больших датасетов)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Регуляризация весов')
    parser.add_argument('--warmup_steps', type=int, default=50, help='Шаги разогрева (уменьшено для дообучения)')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--resume_training', action='store_true', help='Дообучение существующей модели')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)

if __name__ == '__main__':
    main() 