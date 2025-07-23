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
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
from loguru import logger

class LegalDataset(Dataset):
    """
    Датасет для обучения модели на парах "факты → мотивировка"
    """
    def __init__(self, file_path, tokenizer, max_input_length=1024, max_output_length=1024):
        """
        Инициализация датасета
        
        Args:
            file_path: Путь к JSONL файлу с данными
            tokenizer: Токенизатор из библиотеки transformers
            max_input_length: Максимальная длина входной последовательности
            max_output_length: Максимальная длина выходной последовательности
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Загрузка данных из JSONL файла
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    if 'facts' in example and 'reasoning' in example:
                        self.examples.append(example)
                except json.JSONDecodeError:
                    logger.warning(f"Ошибка декодирования JSON в строке: {line[:50]}...")
                    continue
        
        logger.info(f"Загружено {len(self.examples)} примеров из {file_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Токенизация входных и выходных данных
        inputs = self.tokenizer(
            example['facts'], 
            max_length=self.max_input_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                example['reasoning'],
                max_length=self.max_output_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Преобразование тензоров и подготовка возвращаемого словаря
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = labels['input_ids'].squeeze()
        
        # Замена паддинга в метках на -100 (игнорируется при вычислении потерь)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def process_dataset_for_hf_trainer(dataset_path, tokenizer, max_input_length=1024, max_output_length=1024):
    """
    Загрузка и обработка датасета для HF Trainer
    
    Args:
        dataset_path: Путь к JSONL файлу с данными
        tokenizer: Токенизатор модели
        max_input_length: Максимальная длина входной последовательности
        max_output_length: Максимальная длина выходной последовательности
    """
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
                logger.warning(f"Ошибка декодирования JSON в строке: {line[:50]}...")
                continue
    
    # Преобразование в формат HuggingFace Dataset
    hf_dataset = HFDataset.from_list(data)
    
    def preprocess_function(examples):
        inputs = [text for text in examples['facts']]
        targets = [text for text in examples['reasoning']]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=max_input_length, 
            padding='max_length',
            truncation=True
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=max_output_length, 
                padding='max_length',
                truncation=True
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Применение предобработки к датасету
    processed_dataset = hf_dataset.map(
        preprocess_function,
        batched=True,
        desc="Обработка датасета"
    )
    
    return processed_dataset

def train_model(args):
    """
    Обучение модели на датасете "факты → мотивировка"
    
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
    else:
        logger.warning("CUDA недоступна! Проверьте:")
        logger.warning("1. Установлен ли CUDA toolkit")
        logger.warning("2. Установлена ли PyTorch с поддержкой CUDA")
        logger.warning("3. Есть ли совместимая видеокарта NVIDIA")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")
    
    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Принудительная загрузка на GPU если доступен
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name
        ).cuda()
        logger.info(f"Модель загружена на GPU: {torch.cuda.get_device_name()}")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name
        ).cpu()
        logger.info("Модель загружена на CPU")
    
    # Если токенизатор не имеет pad_token, используем eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка и обработка датасета
    train_dataset_path = args.train_file
    val_dataset_path = args.test_file
    
    train_dataset = process_dataset_for_hf_trainer(train_dataset_path, tokenizer, args.max_input_length, args.max_output_length)
    val_dataset = process_dataset_for_hf_trainer(val_dataset_path, tokenizer, args.max_input_length, args.max_output_length)
    
    # Создание data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest'
    )
    
    # Настройка аргументов обучения
    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        fp16=False,
        report_to=["tensorboard"],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        dataloader_pin_memory=True,
        dataloader_num_workers=0
    )
    
    # Создание тренера
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Обучение модели
    logger.info("Начало обучения модели...")
    trainer.train()
    
    # Сохранение обученной модели
    logger.info(f"Сохранение модели в {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Обучение завершено!")

def main():
    parser = argparse.ArgumentParser(description='Обучение модели для генерации мотивировочной части решения суда')
    
    # Пути к файлам
    parser.add_argument('--train_file', type=str, required=True, help='Путь к файлу с обучающими данными (JSONL)')
    parser.add_argument('--test_file', type=str, required=True, help='Путь к файлу с тестовыми данными (JSONL)')
    parser.add_argument('--output_dir', type=str, required=True, help='Директория для сохранения модели')
    
    # Параметры модели
    parser.add_argument('--model_name', type=str, default='models/legal_model', help='Название или путь к предобученной модели (по умолчанию использует QVikhr)')
    parser.add_argument('--max_input_length', type=int, default=512, help='Максимальная длина входной последовательности')
    parser.add_argument('--max_output_length', type=int, default=512, help='Максимальная длина выходной последовательности')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=20, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=2, help='Размер батча')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Скорость обучения')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Количество шагов разогрева оптимизатора')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Параметр регуляризации весов')
    parser.add_argument('--logging_steps', type=int, default=10, help='Частота логирования')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Шаги накопления градиента')
    parser.add_argument('--seed', type=int, default=42, help='Seed для генератора случайных чисел')
    
    args = parser.parse_args()
    
    # Создание директории для сохранения модели, если она не существует
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_model(args)

if __name__ == '__main__':
    main() 