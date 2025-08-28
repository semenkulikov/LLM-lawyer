#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from loguru import logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Загрузка конфигурации"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dataset(dataset_path: str) -> Dataset:
    """Загрузка датасета"""
    logger.info(f"Загружаю датасет: {dataset_path}")
    
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    logger.info(f"Загружено {len(examples)} примеров")
    return Dataset.from_list(examples)

def format_prompt(example: Dict[str, str]) -> str:
    """Форматирование промпта для обучения"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Формат для инструкционного обучения
    prompt = f"### Инструкция:\n{instruction}\n\n### Входные данные:\n{input_text}\n\n### Ответ:\n{output}"
    
    return prompt

def tokenize_function(examples: Dict[str, Any], tokenizer, max_length: int = 2048) -> Dict[str, Any]:
    """Токенизация данных"""
    prompts = [format_prompt(ex) for ex in examples]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Устанавливаем labels равными input_ids для causal LM
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Настройка конфигурации LoRA"""
    lora_config = config.get('lora_config', {})
    
    return LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        bias=lora_config.get('bias', "none"),
        task_type=TaskType.CAUSAL_LM
    )

def train_model(config_path: str):
    """Обучение модели с LoRA"""
    logger.info("🚀 Начинаю обучение модели с LoRA")
    
    # Загружаем конфигурацию
    config = load_config(config_path)
    
    # Проверяем CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используется устройство: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Загружаем токенизатор
    model_name = config['model_name']
    logger.info(f"Загружаю токенизатор: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Добавляем pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загружаем модель
    logger.info(f"Загружаю модель: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # Настраиваем LoRA
    logger.info("Настраиваю LoRA")
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    # Выводим информацию о параметрах
    model.print_trainable_parameters()
    
    # Загружаем датасет
    dataset = load_dataset(config['dataset_path'])
    
    # Токенизируем датасет
    logger.info("Токенизирую датасет")
    data_config = config.get('data_config', {})
    max_length = data_config.get('max_length', 2048)
    
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Настройки обучения
    training_args = TrainingArguments(
        **config['training_args'],
        report_to=None,  # Отключаем wandb
        dataloader_pin_memory=False if device == "cpu" else True
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Создаем тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Обучаем модель
    logger.info("Начинаю обучение...")
    trainer.train()
    
    # Сохраняем модель
    output_dir = config['training_args']['output_dir']
    logger.info(f"Сохраняю модель в: {output_dir}")
    
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("✅ Обучение завершено!")
    logger.info(f"Модель сохранена в: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Обучение модели с LoRA")
    parser.add_argument("--config", type=str, required=True, help="Путь к конфигурационному файлу")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Конфигурационный файл не найден: {args.config}")
        return
    
    try:
        train_model(args.config)
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        raise

if __name__ == "__main__":
    main()
