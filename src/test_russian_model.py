#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

def load_model(model_path):
    """
    Загрузка предобученной модели и токенизатора
    """
    logger.info(f"Загрузка модели из {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True)
        
        # Переносим модель на доступное устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Модель успешно загружена и перенесена на {device}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

def generate_text(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """
    Генерация текста с помощью русскоязычной модели
    """
    device = model.device
    
    # Токенизация промпта
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Генерация
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Декодирование результата
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Убираем исходный промпт из результата
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def evaluate_example(model_path, test_file, output_dir, seed=42):
    """
    Оценка модели на случайном примере из тестового датасета
    """
    # Установка seed для воспроизводимости
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Загрузка модели
    model, tokenizer = load_model(model_path)
    
    # Загрузка тестового датасета
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    # Выбор случайного примера
    example = random.choice(test_data)
    facts = example['facts']
    reference = example['reasoning']
    
    logger.info("Выбран случайный пример для оценки")
    logger.info(f"Факты (первые 100 символов): {facts[:100]}...")
    
    # Создание промпта для русской модели
    prompt = f"Фактические обстоятельства дела: {facts}\n\nМотивировочная часть решения суда:"
    
    logger.info("Генерация мотивировочной части...")
    
    # Генерация текста
    generated = generate_text(model, tokenizer, prompt)
    
    # Вывод результатов
    print("\n" + "="*80)
    print("ФАКТЫ:")
    print("-"*80)
    try:
        print(facts)
    except UnicodeEncodeError:
        print(facts.encode('utf-8', errors='replace').decode('utf-8'))
    
    print("\n" + "="*80)
    print("СГЕНЕРИРОВАННАЯ МОТИВИРОВКА:")
    print("-"*80)
    try:
        print(generated)
    except UnicodeEncodeError:
        print(generated.encode('utf-8', errors='replace').decode('utf-8'))
    
    print("\n" + "="*80)
    print("ЭТАЛОННАЯ МОТИВИРОВКА:")
    print("-"*80)
    try:
        print(reference)
    except UnicodeEncodeError:
        print(reference.encode('utf-8', errors='replace').decode('utf-8'))
    
    print("\n" + "="*80)
    
    # Сохранение результатов
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"test_russian_model_{timestamp}.json")
    
    results = {
        'model_path': model_path,
        'test_file': test_file,
        'timestamp': timestamp,
        'facts': facts,
        'generated_reasoning': generated,
        'reference_reasoning': reference,
        'prompt': prompt
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты сохранены в файл {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Тестирование русскоязычной модели')
    parser.add_argument('--model_path', type=str, required=True, help='Путь к модели')
    parser.add_argument('--test_file', type=str, required=True, help='Путь к тестовому файлу')
    parser.add_argument('--output_dir', type=str, default='results', help='Директория для сохранения результатов')
    parser.add_argument('--seed', type=int, default=42, help='Seed для воспроизводимости')
    
    args = parser.parse_args()
    
    evaluate_example(args.model_path, args.test_file, args.output_dir, args.seed)

if __name__ == '__main__':
    from datetime import datetime
    main() 