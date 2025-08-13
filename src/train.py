#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
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

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        # Убеждаемся, что все токены в допустимых пределах
        tokenized["input_ids"] = torch.clamp(tokenized["input_ids"], 0, tokenizer.vocab_size - 1)
        tokenized["labels"] = tokenized["input_ids"].clone()
        # Преобразуем тензоры в списки — чтобы HF .map вернул корректный формат для Trainer
        return {k: v.numpy() for k, v in tokenized.items()}

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
    """Загрузка существующей модели для дообучения (без предварительного float16)"""
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "config.json")):
        logger.info(f"Загрузка существующей модели из {output_dir}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                output_dir,
                low_cpu_mem_usage=True,
            )
            logger.info("Существующая модель загружена успешно")
            return model
        except Exception as e:
            logger.warning(f"Не удалось загрузить существующую модель: {e}")
            logger.info("Загружаем базовую модель для обучения с нуля")

    logger.info(f"Загрузка базовой модели {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
    )
    return model

def train_model(args):
    """Обучение модели с использованием bf16, gradient_checkpointing и безопасной конфигурацией"""
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
            if gpu_memory < 8:
                logger.warning(f"GPU {i} имеет мало памяти ({gpu_memory:.1f} GB). Рекомендуется 8GB+ для стабильной работы")
            else:
                logger.info(f"✅ GPU {i} имеет достаточно памяти ({gpu_memory:.1f} GB)")
    else:
        logger.warning("CUDA недоступна! Обучение будет происходить на CPU (медленнее)")

    # Загружаем токенизатор
    logger.info("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Размер словаря токенизатора: {tokenizer.vocab_size}")

    # Создаём резервную копию при resume
    if args.resume_training:
        backup_previous_model(args.output_dir)

    # Проверяем поддержку bf16
    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    logger.info(f"Поддержка bf16: {bf16_available}")

    # Загружаем модель (без явного float16)
    model = load_existing_model(args.model_name, args.output_dir, tokenizer)

    # Отключаем use_cache (важно при gradient checkpointing)
    try:
        model.config.use_cache = False
    except Exception:
        # Некоторые модели могут не иметь config.use_cache
        pass

    # Включаем gradient checkpointing для экономии памяти
    model.gradient_checkpointing_enable()

    # Перемещаем модель на устройство и переводим в bfloat16, если поддерживается
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bf16_available and device.type == "cuda":
        try:
            model = model.to(device).to(torch.bfloat16)
            logger.info("Модель переведена в bfloat16 на CUDA")
        except Exception as e:
            logger.warning(f"Не удалось привести модель к bfloat16: {e}. Оставляем в FP32 на CUDA")
            model = model.to(device)
    else:
        model = model.to(device)

    # Очищаем CUDA кеш для освобождения памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"CUDA кеш очищен. Доступная память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Обрабатываем датасеты
    effective_max_length = args.max_length
    logger.info(f"Используем max_length: {effective_max_length}")

    train_dataset = process_dataset_for_hf_trainer(args.train_file, tokenizer, effective_max_length)
    val_dataset = process_dataset_for_hf_trainer(args.test_file, tokenizer, effective_max_length)

    # TrainingArguments — используем bf16, отключаем fp16
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=10,
        save_steps=200,
        eval_steps=100,
        save_total_limit=3,
        seed=args.seed,

        # Mixed precision: bf16 = True if поддерживается
        fp16=False,
        bf16=bf16_available,

        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,

        dataloader_pin_memory=True,
        remove_unused_columns=True,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,

        report_to=[],
        logging_dir=f"{args.output_dir}/logs",

        save_strategy="steps",
        eval_strategy="steps",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=2,
        group_by_length=True,

        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",

        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Обучаем
    logger.info("Начало обучения...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint if hasattr(args, "resume_from_checkpoint") else None)


    # Сохраняем модель и токенизатор
    logger.info(f"Сохранение модели в {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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
    parser = argparse.ArgumentParser(description='Обучение модели (bf16 путь, gradient checkpointing)')
    parser.add_argument('--train_file', type=str, required=True, help='Путь к обучающему файлу')
    parser.add_argument('--test_file', type=str, required=True, help='Путь к тестовому файлу')
    parser.add_argument('--output_dir', type=str, required=True, help='Директория для сохранения')
    parser.add_argument('--model_name', type=str, default='Vikhrmodels/QVikhr-3-4B-Instruction', help='Модель')
    parser.add_argument('--max_length', type=int, default=1024, help='Максимальная длина последовательности')
    parser.add_argument('--epochs', type=int, default=15, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=2, help='Размер батча')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Шаги накопления градиента')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Скорость обучения')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Регуляризация весов')
    parser.add_argument('--warmup_steps', type=int, default=50, help='Шаги разогрева')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--resume_training', action='store_true', help='Дообучение существующей модели')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Путь к чекпоинту для возобновления')


    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)

if __name__ == '__main__':
    main()
