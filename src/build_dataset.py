import os
import json
import logging
import argparse
import pathlib
from typing import Dict, List, Any
import random
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def clean_text(text: str) -> str:
    """
    Очищает текст от лишних символов и форматирования
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    if not text:
        return ""
    
    # Удаляем лишние пробелы и переносы строк
    text = " ".join(text.split())
    
    return text.strip()

def filter_valid_pairs(facts: str, reasoning: str) -> bool:
    """
    Проверяет, является ли пара "факты → мотивировка" подходящей для обучения
    
    Args:
        facts: Текст с фактами
        reasoning: Текст с мотивировкой
        
    Returns:
        True, если пара подходит для обучения, иначе False
    """
    # Проверяем минимальную длину текстов
    if len(facts) < 100 or len(reasoning) < 200:
        return False
    
    # Проверяем максимальную длину (для избежания слишком длинных текстов)
    if len(facts) > 5000 or len(reasoning) > 10000:
        return False
    
    # Проверяем соотношение длин (мотивировка обычно длиннее фактов)
    if len(reasoning) < len(facts):
        return False
    
    return True

def process_document(file_path: str) -> Dict[str, str]:
    """
    Обрабатывает один документ и извлекает пары "факты → мотивировка"
    
    Args:
        file_path: Путь к JSON-файлу с документом
        
    Returns:
        Словарь с фактами и мотивировкой, или пустой словарь, если пара не подходит
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Извлекаем секции из документа
        sections = data.get('sections', {})
        facts = clean_text(sections.get('facts', ''))
        reasoning = clean_text(sections.get('reasoning', ''))
        
        # Если нет фактов или мотивировки, пропускаем документ
        if not facts or not reasoning:
            return {}
        
        # Проверяем, подходит ли пара для обучения
        if not filter_valid_pairs(facts, reasoning):
            return {}
        
        # Возвращаем пару "факты → мотивировка"
        return {
            "facts": facts,
            "reasoning": reasoning,
            "source": os.path.basename(file_path)
        }
    
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {file_path}: {str(e)}")
        return {}

def build_dataset(input_dir: str, output_file: str, test_split: float = 0.1) -> None:
    """
    Создает обучающий датасет из структурированных данных
    
    Args:
        input_dir: Путь к директории со структурированными данными (JSON)
        output_file: Путь к выходному файлу (JSONL)
        test_split: Доля тестовых данных (от 0 до 1)
    """
    # Проверяем, существует ли входная директория
    if not os.path.exists(input_dir):
        logging.error(f"Директория {input_dir} не существует")
        return
    
    # Получаем список всех JSON-файлов в директории
    json_files = list(pathlib.Path(input_dir).glob('**/*.json'))
    logging.info(f"Найдено {len(json_files)} JSON-файлов в директории {input_dir}")
    
    # Обрабатываем каждый файл и формируем датасет
    dataset = []
    
    for file_path in tqdm(json_files, desc="Обработка файлов"):
        pair = process_document(str(file_path))
        if pair:
            dataset.append(pair)
    
    logging.info(f"Сформировано {len(dataset)} пар 'факты → мотивировка'")
    
    # Перемешиваем датасет для лучшего обучения
    random.shuffle(dataset)
    
    # Разделяем на обучающую и тестовую выборки
    split_idx = int(len(dataset) * (1 - test_split))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    # Создаем директории для выходных файлов, если они не существуют
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Сохраняем обучающую выборку
    train_output = output_file
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Сохраняем тестовую выборку
    test_output = output_file.replace('.jsonl', '_test.jsonl')
    with open(test_output, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logging.info(f"Сохранено {len(train_data)} пар в файл {train_output}")
    logging.info(f"Сохранено {len(test_data)} пар в файл {test_output}")
    
    # Сохраняем метаданные датасета
    meta_output = output_file.replace('.jsonl', '_meta.json')
    meta = {
        "total_pairs": len(dataset),
        "train_pairs": len(train_data),
        "test_pairs": len(test_data),
        "avg_facts_length": sum(len(item["facts"]) for item in dataset) // len(dataset) if dataset else 0,
        "avg_reasoning_length": sum(len(item["reasoning"]) for item in dataset) // len(dataset) if dataset else 0,
    }
    
    with open(meta_output, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Метаданные датасета сохранены в файл {meta_output}")

def main():
    parser = argparse.ArgumentParser(description='Создание обучающего датасета для модели генерации мотивировочной части')
    parser.add_argument('input_dir', help='Директория со структурированными данными (JSON)')
    parser.add_argument('output_file', help='Путь к выходному файлу (JSONL)')
    parser.add_argument('--test-split', type=float, default=0.1, help='Доля тестовых данных (от 0 до 1)')
    
    args = parser.parse_args()
    
    build_dataset(args.input_dir, args.output_file, args.test_split)

if __name__ == "__main__":
    main() 