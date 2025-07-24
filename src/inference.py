#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

def load_model(model_path):
    """
    Загрузка предобученной модели QVikhr-3-4B и токенизатора
    
    Args:
        model_path: Путь к директории с сохраненной моделью
        
    Returns:
        model: Загруженная модель
        tokenizer: Загруженный токенизатор
    """
    logger.info(f"Загрузка модели QVikhr-3-4B из {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Загрузка модели с GPU оптимизациями
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Mixed precision
                device_map="auto",  # Автоматическое распределение
                low_cpu_mem_usage=True,
            )
            logger.info("Модель QVikhr-3-4B загружена на GPU с автоматическим распределением памяти")
        else:
            logger.warning("CUDA недоступна! Модель будет загружена на CPU (медленно)")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            logger.info("Модель загружена на CPU")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

def generate(model, tokenizer, facts, max_input_length=1024, max_output_length=1024, 
             num_beams=4, do_sample=True, temperature=0.7, top_p=0.9):
    """
    Генерация мотивировочной части на основе фактических обстоятельств дела
    
    Args:
        model: Предобученная модель QVikhr-3-4B
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
    # Определяем устройство для входных данных
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Формируем промпт для CausalLM
    prompt = f"Факты: {facts}\nМотивировка:"
    
    # Токенизация входных данных
    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Перемещаем входные данные на правильное устройство
    if device.type == 'cuda':
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Генерация текста с оптимизированными параметрами
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_output_length,  # Используем max_new_tokens
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,  # Увеличенный штраф за повторения
            length_penalty=0.8,      # Штраф за длину
            early_stopping=True,     # Остановка при EOS
            no_repeat_ngram_size=3,  # Запрет повторения n-грамм
        )
    
    # Декодирование результата
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Извлекаем только сгенерированную часть (после "Мотивировка:")
    if "Мотивировка:" in generated_text:
        generated_text = generated_text.split("Мотивировка:")[1].strip()
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Генерация мотивировочной части решения суда с QVikhr-3-4B')
    
    # Аргументы
    parser.add_argument('--model_path', type=str, default='models/legal_model', help='Путь к директории с сохраненной моделью QVikhr-3-4B')
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
    
    # Проверка существования модели
    if not os.path.exists(args.model_path):
        logger.error(f"Модель QVikhr-3-4B не найдена по пути: {args.model_path}")
        logger.error("Убедитесь, что модель обучена и сохранена в указанной директории")
        return
    
    # Загрузка модели и токенизатора
    model, tokenizer = load_model(args.model_path)
    
    # Получение входного текста
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            facts = f.read()
    else:
        facts = args.input_text
    
    # Генерация мотивировочной части
    logger.info("Генерация мотивировочной части с QVikhr-3-4B...")
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
        print("СГЕНЕРИРОВАННАЯ МОТИВИРОВОЧНАЯ ЧАСТЬ (QVikhr-3-4B):\n")
        print(reasoning)
        print("\n" + "="*80)

if __name__ == '__main__':
    main() 