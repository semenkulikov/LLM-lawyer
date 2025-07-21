#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
from pathlib import Path
from loguru import logger

def run_command(cmd, description):
    """Запуск команды с логированием"""
    logger.info(f"Запуск: {description}")
    logger.info(f"Команда: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        logger.info(f"✓ {description} завершено успешно")
        if result.stdout:
            logger.info(f"Вывод: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Ошибка в {description}: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False

def check_dependencies():
    """Проверка зависимостей"""
    logger.info("Проверка зависимостей...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'openai', 
        'pymupdf', 'nltk', 'tqdm', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Отсутствуют пакеты: {', '.join(missing_packages)}")
        logger.info("Установите их командой: pip install -r requirements.txt")
        return False
    
    logger.info("✓ Все зависимости установлены")
    return True

def check_openai_key():
    """Проверка наличия OpenAI API ключа"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("Не установлена переменная окружения OPENAI_API_KEY")
        logger.info("Установите её командой: set OPENAI_API_KEY=your_key_here")
        return False
    
    logger.info("✓ OpenAI API ключ найден")
    return True

def create_directories():
    """Создание необходимых директорий"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/analyzed',
        'data/structured',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Директории созданы")

def step1_preprocess(input_dir=None, output_dir="data/processed"):
    """Шаг 1: Предобработка PDF документов"""
    if not input_dir:
        input_dir = "data/raw"
    
    if not Path(input_dir).exists():
        logger.warning(f"Директория {input_dir} не существует. Пропускаем предобработку.")
        return True
    
    cmd = [
        sys.executable, "src/preprocess.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir
    ]
    
    return run_command(cmd, "Предобработка PDF документов")

def step2_analyze_with_openai(input_dir="data/processed", output_dir="data/analyzed", max_docs=3):
    """Шаг 2: Анализ документов с помощью OpenAI"""
    if not Path(input_dir).exists():
        logger.warning(f"Директория {input_dir} не существует. Пропускаем анализ.")
        return True
    
    cmd = [
        sys.executable, "src/process_with_openai.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--max-docs", str(max_docs)
    ]
    
    return run_command(cmd, "Анализ документов с OpenAI")

def step3_build_dataset(analyzed_dir="data/analyzed", output_file="data/train_dataset.jsonl"):
    """Шаг 3: Создание обучающего датасета"""
    if not Path(analyzed_dir).exists():
        logger.warning(f"Директория {analyzed_dir} не существует. Пропускаем создание датасета.")
        return True
    
    cmd = [
        sys.executable, "src/build_dataset.py",
        "--analyzed-dir", analyzed_dir,
        "--output-file", output_file
    ]
    
    return run_command(cmd, "Создание обучающего датасета")

def step4_train_model(train_file="data/train_dataset.jsonl", 
                     test_file="data/train_dataset_test.jsonl",
                     output_dir="models/legal_model",
                     epochs=3,
                     batch_size=4):
    """Шаг 4: Обучение модели"""
    if not Path(train_file).exists():
        logger.warning(f"Файл {train_file} не существует. Пропускаем обучение.")
        return True
    
    cmd = [
        sys.executable, "src/train.py",
        "--train_file", train_file,
        "--test_file", test_file,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size)
    ]
    
    return run_command(cmd, "Обучение модели")

def step5_test_model(model_path="models/legal_model", 
                    test_file="data/train_dataset_test.jsonl",
                    output_dir="results"):
    """Шаг 5: Тестирование модели"""
    if not Path(model_path).exists():
        logger.warning(f"Модель {model_path} не найдена. Пропускаем тестирование.")
        return True
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "src/test_example.py",
        "--model_path", model_path,
        "--test_file", test_file,
        "--output_dir", output_dir
    ]
    
    return run_command(cmd, "Тестирование модели")

def run_full_pipeline(args):
    """Запуск полного пайплайна"""
    logger.info("=" * 60)
    logger.info("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ОБРАБОТКИ ДОКУМЕНТОВ")
    logger.info("=" * 60)
    
    # Проверки
    if not check_dependencies():
        return False
    
    if not check_openai_key():
        return False
    
    create_directories()
    
    # Выполнение шагов
    steps = [
        ("Предобработка PDF", lambda: step1_preprocess(args.input_dir)),
        ("Анализ с OpenAI", lambda: step2_analyze_with_openai(max_docs=args.max_docs)),
        ("Создание датасета", step3_build_dataset),
        ("Обучение модели", lambda: step4_train_model(epochs=args.epochs, batch_size=args.batch_size)),
        ("Тестирование модели", step5_test_model)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n--- Шаг: {step_name} ---")
        if not step_func():
            logger.error(f"Пайплайн остановлен на шаге: {step_name}")
            return False
    
    logger.info("\n" + "=" * 60)
    logger.info("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
    logger.info("=" * 60)
    logger.info("Для запуска GUI используйте: python gui/app.py")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Полный пайплайн обработки юридических документов')
    parser.add_argument('--input-dir', type=str, help='Директория с исходными PDF файлами')
    parser.add_argument('--max-docs', type=int, default=3, help='Максимальное количество документов для анализа')
    parser.add_argument('--epochs', type=int, default=3, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=4, help='Размер батча для обучения')
    parser.add_argument('--skip-training', action='store_true', help='Пропустить обучение модели')
    
    args = parser.parse_args()
    
    if args.skip_training:
        # Только предобработка и анализ
        logger.info("Запуск без обучения модели...")
        check_dependencies()
        check_openai_key()
        create_directories()
        
        step1_preprocess(args.input_dir)
        step2_analyze_with_openai(max_docs=args.max_docs)
        step3_build_dataset()
        
        logger.info("Предобработка завершена. Для обучения запустите: python src/train.py")
    else:
        # Полный пайплайн
        run_full_pipeline(args)

if __name__ == '__main__':
    main() 