#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Загрузка предобученной модели и токенизатора
    
    Args:
        model_path: Путь к директории с сохраненной моделью
        
    Returns:
        model: Загруженная модель
        tokenizer: Загруженный токенизатор
    """
    logger.info(f"Загрузка модели из {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Переносим модель на доступное устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Модель успешно загружена и перенесена на {device}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

def generate(model, tokenizer, facts, max_input_length=1024, max_output_length=1024, 
             num_beams=4, do_sample=True, temperature=0.7, top_p=0.9):
    """
    Генерация мотивировочной части на основе фактических обстоятельств дела
    
    Args:
        model: Предобученная модель
        tokenizer: Токенизатор для модели
        facts: Текст с фактическими обстоятельствами дела
        max_input_length: Максимальная длина входной последовательности
        max_output_length: Максимальная длина выходной последовательности
        num_beams: Количество лучей для beam search
        do_sample: Использовать ли семплирование
        temperature: Температура для семплирования
        top_p: Параметр top-p для семплирования
        
    Returns:
        generated_text: Сгенерированная мотивировочная часть
    """
    device = model.device
    
    # Токенизация входных данных
    inputs = tokenizer(
        facts,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Генерация текста
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Декодирование результата
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Генерация мотивировочной части решения суда')
    
    # Аргументы
    parser.add_argument('--model_path', type=str, required=True, help='Путь к директории с сохраненной моделью')
    parser.add_argument('--input_text', type=str, help='Текст с фактическими обстоятельствами дела')
    parser.add_argument('--input_file', type=str, help='Файл с фактическими обстоятельствами дела')
    parser.add_argument('--output_file', type=str, help='Файл для сохранения сгенерированной мотивировки')
    parser.add_argument('--max_input_length', type=int, default=1024, help='Максимальная длина входной последовательности')
    parser.add_argument('--max_output_length', type=int, default=1024, help='Максимальная длина выходной последовательности')
    parser.add_argument('--num_beams', type=int, default=4, help='Количество лучей для beam search')
    parser.add_argument('--do_sample', action='store_true', help='Использовать семплирование')
    parser.add_argument('--temperature', type=float, default=0.7, help='Температура для семплирования')
    parser.add_argument('--top_p', type=float, default=0.9, help='Параметр top-p для семплирования')
    
    args = parser.parse_args()
    
    # Проверка параметров
    if not args.input_text and not args.input_file:
        parser.error("Необходимо указать либо --input_text, либо --input_file")
    
    # Загрузка модели и токенизатора
    model, tokenizer = load_model(args.model_path)
    
    # Получение входного текста
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            facts = f.read()
    else:
        facts = args.input_text
    
    # Генерация мотивировочной части
    logger.info("Генерация мотивировочной части...")
    reasoning = generate(
        model=model,
        tokenizer=tokenizer,
        facts=facts,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Вывод результата
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(reasoning)
        logger.info(f"Результат сохранен в файл {args.output_file}")
    else:
        print("\n" + "="*80 + "\n")
        print("СГЕНЕРИРОВАННАЯ МОТИВИРОВОЧНАЯ ЧАСТЬ:\n")
        print(reasoning)
        print("\n" + "="*80)

if __name__ == '__main__':
    main() 