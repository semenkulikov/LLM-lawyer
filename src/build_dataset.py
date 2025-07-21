#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger

def load_analyzed_documents(analyzed_dir: str) -> List[Dict[str, Any]]:
    """
    Загружает проанализированные документы из директории
    
    Args:
        analyzed_dir: Директория с JSON файлами анализа
        
    Returns:
        Список словарей с данными документов
    """
    documents = []
    analyzed_path = Path(analyzed_dir)
    
    if not analyzed_path.exists():
        logger.error(f"Директория {analyzed_dir} не существует")
        return documents
    
    # Ищем все JSON файлы с результатами анализа
    json_files = list(analyzed_path.glob("*_analyzed.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.append(data)
                logger.info(f"Загружен документ: {json_file.name}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке {json_file}: {e}")
    
    logger.info(f"Всего загружено документов: {len(documents)}")
    return documents

def create_training_examples(documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Создает обучающие примеры из проанализированных документов
    
    Args:
        documents: Список проанализированных документов
        
    Returns:
        Список обучающих примеров в формате {"facts": "...", "reasoning": "..."}
    """
    examples = []
    
    for doc in documents:
        try:
            # Извлекаем факты из установочной части
            facts = doc.get('sections', {}).get('facts', '')
            if not facts:
                facts = doc.get('document_info', {}).get('key_facts', '')
            
            # Извлекаем мотивировку
            reasoning = doc.get('sections', {}).get('reasoning', '')
            if not reasoning:
                reasoning = doc.get('document_info', {}).get('legal_norms', '')
            
            # Проверяем, что у нас есть и факты, и мотивировка
            if facts and reasoning and len(facts) > 50 and len(reasoning) > 50:
                example = {
                    "facts": facts.strip(),
                    "reasoning": reasoning.strip()
                }
                examples.append(example)
                logger.info(f"Создан пример из документа: {doc.get('document_info', {}).get('filename', 'unknown')}")
            else:
                logger.warning(f"Пропущен документ {doc.get('document_info', {}).get('filename', 'unknown')} - недостаточно данных")
                
        except Exception as e:
            logger.error(f"Ошибка при создании примера: {e}")
            continue
    
    logger.info(f"Создано обучающих примеров: {len(examples)}")
    return examples

def save_dataset(examples: List[Dict[str, str]], output_file: str, test_ratio: float = 0.2):
    """
    Сохраняет датасет в формате JSONL с разделением на train/test
    
    Args:
        examples: Список обучающих примеров
        output_file: Путь к основному файлу датасета
        test_ratio: Доля данных для тестирования
    """
    if not examples:
        logger.error("Нет примеров для сохранения")
        return
    
    # Разделяем на train и test
    split_idx = int(len(examples) * (1 - test_ratio))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    # Сохраняем основной датасет
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Сохраняем тестовый датасет
    test_file = output_path.parent / f"{output_path.stem}_test.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Сохраняем метаданные
    meta_file = output_path.parent / f"{output_path.stem}_meta.json"
    metadata = {
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "test_ratio": test_ratio,
        "output_file": str(output_path),
        "test_file": str(test_file)
    }
    
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Датасет сохранен:")
    logger.info(f"  Train: {output_path} ({len(train_examples)} примеров)")
    logger.info(f"  Test: {test_file} ({len(test_examples)} примеров)")
    logger.info(f"  Metadata: {meta_file}")

def main():
    parser = argparse.ArgumentParser(description='Создание обучающего датасета из проанализированных документов')
    parser.add_argument('--analyzed-dir', type=str, required=True, 
                       help='Директория с проанализированными документами (JSON файлы)')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Путь к выходному файлу датасета (JSONL)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Доля данных для тестирования (по умолчанию 0.2)')
    
    args = parser.parse_args()
    
    # Загружаем проанализированные документы
    documents = load_analyzed_documents(args.analyzed_dir)
    
    if not documents:
        logger.error("Не найдено проанализированных документов")
        return
    
    # Создаем обучающие примеры
    examples = create_training_examples(documents)
    
    if not examples:
        logger.error("Не удалось создать обучающие примеры")
        return
    
    # Сохраняем датасет
    save_dataset(examples, args.output_file, args.test_ratio)
    
    logger.info("Создание датасета завершено успешно!")

if __name__ == '__main__':
    main() 