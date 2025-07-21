#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
import argparse
from inference import load_model, generate
from loguru import logger

def get_random_example(test_file, seed=None):
    """
    Получить случайный пример из тестового файла
    
    Args:
        test_file: Путь к тестовому файлу (JSONL)
        seed: Seed для генератора случайных чисел
        
    Returns:
        example: Случайный пример из файла
    """
    if seed is not None:
        random.seed(seed)
    
    examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line)
                if 'facts' in example and 'reasoning' in example:
                    examples.append(example)
            except json.JSONDecodeError:
                continue
    
    if not examples:
        raise ValueError(f"В файле {test_file} не найдено примеров")
    
    return random.choice(examples)

def evaluate_example(model_path, test_file, output_dir=None, seed=None):
    """
    Оценить модель на случайном примере
    
    Args:
        model_path: Путь к директории с моделью
        test_file: Путь к тестовому файлу (JSONL)
        output_dir: Директория для сохранения результатов
        seed: Seed для генератора случайных чисел
    """
    # Загрузка модели
    model, tokenizer = load_model(model_path)
    
    # Получение случайного примера
    example = get_random_example(test_file, seed)
    facts = example['facts']
    reference = example['reasoning']
    
    logger.info("Выбран случайный пример для оценки")
    logger.info(f"Факты (первые 100 символов): {facts[:100]}...")
    
    # Генерация мотивировочной части
    logger.info("Генерация мотивировочной части...")
    generated = generate(model, tokenizer, facts)
    
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
    
    # Сохранение результатов, если указана директория
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"test_example_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'facts': facts,
                'generated': generated,
                'reference': reference
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Результаты сохранены в файл {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Проверка модели на примере')
    parser.add_argument('--model_path', type=str, default='models/legal_model', help='Путь к директории с моделью (по умолчанию использует QVikhr)')
    parser.add_argument('--test_file', type=str, required=True, help='Путь к тестовому файлу (JSONL)')
    parser.add_argument('--output_dir', type=str, help='Директория для сохранения результатов')
    parser.add_argument('--seed', type=int, help='Seed для генератора случайных чисел')
    
    args = parser.parse_args()
    
    evaluate_example(args.model_path, args.test_file, args.output_dir, args.seed)

if __name__ == '__main__':
    main() 