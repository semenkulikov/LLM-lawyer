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

def process_dataset_for_qvikhr(dataset_path, tokenizer, max_length=2048):
    """
    Загрузка и обработка датасета для QVikhr модели
    
    Args:
        dataset_path: Путь к JSONL файлу с данными
        tokenizer: Токенизатор модели
        max_length: Максимальная длина последовательности
    """
    examples = []
    
    # Загрузка данных из JSONL файла
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line)
                if 'facts' in example and 'reasoning' in example:
                    examples.append(example)
            except json.JSONDecodeError:
                logger.warning(f"Ошибка декодирования JSON в строке: {line[:50]}...")
                continue
    
    logger.info(f"Загружено {len(examples)} примеров из {dataset_path}")
    
    def preprocess_function(examples):
        """
        Предобработка данных для QVikhr
        """
        # Создаем промпты в формате инструкций
        prompts = []
        for i in range(len(examples['facts'])):
            prompt = f"""<|im_start|>system
Ты - опытный юрист, специализирующийся на составлении мотивировочных частей судебных решений. 
Твоя задача - на основе фактических обстоятельств дела составить грамотную мотивировочную часть решения суда.
Используй юридическую терминологию и ссылки на нормы права.
<|im_end|>
<|im_start|>user
Фактические обстоятельства дела: {examples['facts'][i]}

Составь мотивировочную часть решения суда на основе этих фактов.
<|im_end|>
<|im_start|>assistant
{examples['reasoning'][i]}
<|im_end|>"""
            prompts.append(prompt)
        
        # Токенизация
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Устанавливаем labels равными input_ids для causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Создаем HuggingFace датасет
    dataset_dict = {
        'facts': [ex['facts'] for ex in examples],
        'reasoning': [ex['reasoning'] for ex in examples]
    }
    
    dataset = HFDataset.from_dict(dataset_dict)
    
    # Применяем предобработку
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Обработка датасета"
    )
    
    return processed_dataset

def train_qvikhr(args):
    """
    Дообучение QVikhr модели на датасете "факты → мотивировка"
    
    Args:
        args: Аргументы командной строки
    """
    # Настройка seed для воспроизводимости результатов
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # Проверка доступности CUDA
    logger.info(f"PyTorch версия: {torch.__version__}")
    logger.info(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA версия: {torch.version.cuda}")
        logger.info(f"Количество GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Загрузка токенизатора и модели
    logger.info("Загрузка QVikhr модели...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_safetensors=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        use_safetensors=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Если токенизатор не имеет pad_token, используем eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка и обработка датасета
    train_dataset_path = args.train_file
    val_dataset_path = args.test_file
    
    logger.info("Обработка обучающего датасета...")
    train_dataset = process_dataset_for_qvikhr(train_dataset_path, tokenizer, args.max_length)
    
    logger.info("Обработка валидационного датасета...")
    val_dataset = process_dataset_for_qvikhr(val_dataset_path, tokenizer, args.max_length)
    
    # Создание data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Для causal LM
    )
    
    # Настройка аргументов обучения
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        eval_steps=args.logging_steps,
        save_steps=args.logging_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=False
    )
    
    # Создание тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Обучение модели
    logger.info("Начало дообучения QVikhr модели...")
    trainer.train()
    
    # Сохранение обученной модели
    logger.info(f"Сохранение модели в {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Дообучение завершено!")

def main():
    parser = argparse.ArgumentParser(description='Дообучение QVikhr модели для генерации мотивировочной части решения суда')
    
    # Пути к файлам
    parser.add_argument('--train_file', type=str, required=True, help='Путь к файлу с обучающими данными (JSONL)')
    parser.add_argument('--test_file', type=str, required=True, help='Путь к файлу с тестовыми данными (JSONL)')
    parser.add_argument('--output_dir', type=str, required=True, help='Директория для сохранения обученной модели')
    
    # Параметры модели
    parser.add_argument('--model_name', type=str, default='Vikhrmodels/QVikhr-3-4B-Instruction', help='Название или путь к предобученной модели')
    parser.add_argument('--max_length', type=int, default=2048, help='Максимальная длина последовательности')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=3, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=1, help='Размер батча')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Скорость обучения')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Количество шагов разогрева оптимизатора')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Параметр регуляризации весов')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Количество шагов накопления градиентов')
    parser.add_argument('--logging_steps', type=int, default=10, help='Частота логирования')
    parser.add_argument('--seed', type=int, default=42, help='Seed для воспроизводимости результатов')
    
    args = parser.parse_args()
    
    train_qvikhr(args)

if __name__ == '__main__':
    main() 