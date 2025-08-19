#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import numpy as np
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

# --- Быстрые флаги ускорения на Ampere/3070 ---
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass


def process_dataset_for_hf_trainer(dataset_path, tokenizer, max_length=2048):
    """Обработка датасета для HF Trainer."""
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

        # Важно: НЕ создаём лишние torch-тензоры здесь.
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            # return_tensors=None -> вернёт списки/np, коллатор преобразует в torch на лету
        )

        # Метки = сдвиг/копия входов для CausalLM
        # (для простоты — полная копия input_ids; паддинг-токены Trainer замаскирует)
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    processed_dataset = hf_dataset.map(
        preprocess_function,
        batched=True,
        desc="Обработка датасета",
        remove_columns=hf_dataset.column_names
    )
    return processed_dataset


def backup_previous_model(output_dir):
    """Создание резервной копии предыдущей модели."""
    if os.path.exists(output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{output_dir}_backup_{timestamp}"
        logger.info(f"Создание резервной копии: {backup_dir}")
        shutil.copytree(output_dir, backup_dir)
        return backup_dir
    return None


def load_existing_model(model_name, output_dir, tokenizer):
    """Загрузка существующей модели для дообучения."""
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


class DebugTrainer(Trainer):
    """Trainer, который один раз логирует устройства батча и модели."""
    def training_step(self, model, inputs, num_items_in_batch=None):
        if not hasattr(self, "_devices_logged"):
            try:
                model_dev = next(model.parameters()).device
                batch_devs = {k: (v.device if isinstance(v, torch.Tensor) else type(v)) for k, v in inputs.items()}
                logger.info(f"[DEBUG] Model device: {model_dev}; Batch devices: {batch_devs}")
            except Exception as e:
                logger.warning(f"[DEBUG] Не удалось вывести устройства: {e}")
            self._devices_logged = True
        return super().training_step(model, inputs, num_items_in_batch)


def train_model(args):
    """Обучение модели (bf16 + gradient_checkpointing) с диагностикой устройств."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(args.cuda_device)
        except Exception:
            pass

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

    # Токенизатор
    logger.info("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Размер словаря токенизатора: {tokenizer.vocab_size}")

    # Резервная копия при резюме
    if args.resume_training:
        backup_previous_model(args.output_dir)

    # Поддержка bf16
    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    logger.info(f"Поддержка bf16: {bf16_available}")

    # Модель
    model = load_existing_model(args.model_name, args.output_dir, tokenizer)

    # Отключаем use_cache (важно при gradient checkpointing)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Gradient checkpointing
    model.gradient_checkpointing_enable()

    # На устройство
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

    # Очистка кеша
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"CUDA кеш очищен. Видеопамять (теоретическая): {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Датасеты
    effective_max_length = args.max_length
    logger.info(f"Используем max_length: {effective_max_length}")
    train_dataset = process_dataset_for_hf_trainer(args.train_file, tokenizer, effective_max_length)
    val_dataset = process_dataset_for_hf_trainer(args.test_file, tokenizer, effective_max_length)

    # Аргументы обучения
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

        fp16=False,
        bf16=bf16_available,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,

        dataloader_pin_memory=args.pin_memory,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor,

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

        group_by_length=args.group_by_length,

        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",

        ddp_find_unused_parameters=False,

        # Позволяет быстро проверить пайплайн без эпох:
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
    )

    # Trainer с диагностикой
    trainer_cls = DebugTrainer if args.log_devices else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Обучаем
    logger.info("Начало обучения...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint if hasattr(args, "resume_from_checkpoint") else None)

    # Сохраняем
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
    parser = argparse.ArgumentParser(description='Обучение модели (bf16 + gradient checkpointing) с диагностикой устройств')
    parser.add_argument('--train_file', type=str, required=True, help='Путь к обучающему файлу')
    parser.add_argument('--test_file', type=str, required=True, help='Путь к тестовому файлу')
    parser.add_argument('--output_dir', type=str, required=True, help='Директория для сохранения')
    parser.add_argument('--model_name', type=str, default='Vikhrmodels/QVikhr-3-4B-Instruction', help='Базовая модель')
    parser.add_argument('--max_length', type=int, default=1024, help='Максимальная длина последовательности')
    parser.add_argument('--epochs', type=int, default=15, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=2, help='Размер микробатча на устройство')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Шаги накопления градиента')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Скорость обучения')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Регуляризация весов')
    parser.add_argument('--warmup_steps', type=int, default=50, help='Шаги разогрева')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--resume_training', action='store_true', help='Создать бэкап и продолжить обучение')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Путь к чекпоинту для возобновления')

    # Новые полезные флаги
    parser.add_argument('--num_workers', type=int, default=min(os.cpu_count() or 2, 4), help='DataLoader workers (Windows часто 2-4 оптимально)')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor (работает при num_workers>0)')
    parser.add_argument('--cuda_device', type=int, default=0, help='Какой GPU использовать (по умолчанию 0)')
    parser.add_argument('--log_devices', action='store_true', help='Вывести устройства модели и батча в первом шаге')
    parser.add_argument('--max_steps', type=int, default=-1, help='Ограничить число шагов тренировки (для быстрых тестов)')

    # pin_memory: по умолчанию ВКЛЮЧЕН, можно отключить --no_pin_memory
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true')
    parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
    parser.set_defaults(pin_memory=True)

    # group_by_length: по умолчанию ВЫКЛ (так как уже паддинг до max_length)
    parser.add_argument('--group_by_length', dest='group_by_length', action='store_true')
    parser.set_defaults(group_by_length=False)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)


if __name__ == '__main__':
    main()
