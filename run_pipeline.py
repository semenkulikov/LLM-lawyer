#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from loguru import logger

# Загружаем переменные окружения из .env файла
def load_env():
    """Загрузка переменных окружения из .env файла"""
    env_file = Path('.env')
    if env_file.exists():
        logger.info("📄 Загрузка переменных окружения из .env файла...")
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Убираем кавычки из значения
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        logger.success("✅ Переменные окружения загружены")
    else:
        logger.warning("⚠️  Файл .env не найден")

# Загружаем переменные окружения при запуске
load_env()

def run_command(cmd, description):
    """Запуск команды с логированием"""
    logger.info(f"🚀 Запуск: {description}")
    logger.info(f"💻 Команда: {' '.join(cmd)}")
    logger.info(f"⏱️  Начало выполнения: {time.strftime('%H:%M:%S')}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ {description} завершено успешно за {elapsed_time:.2f} секунд")
        if result.stdout:
            # Показываем только последние строки вывода
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                logger.info(f"📄 Последние строки вывода:")
                for line in lines[-10:]:
                    logger.info(f"   {line}")
            else:
                logger.info(f"📄 Вывод: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ошибка в {description}: {e}")
        if e.stdout:
            logger.error(f"📄 stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"📄 stderr: {e.stderr}")
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

def step2_analyze_with_openai(input_dir="data/processed", output_dir="data/analyzed", max_docs=50000, max_workers=10):
    """Шаг 2: Анализ документов с OpenAI (оптимизированный)"""
    if not Path(input_dir).exists():
        logger.warning(f"Директория {input_dir} не существует. Пропускаем анализ.")
        return True
    
    cmd = [
        sys.executable, "src/process_with_openai.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--max-docs", str(max_docs),
        "--max-workers", str(max_workers)
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
                     epochs=50,
                     batch_size=8):
    """Шаг 4: Обучение модели (оптимизированное)"""
    if not Path(train_file).exists():
        logger.warning(f"Файл {train_file} не существует. Пропускаем обучение.")
        return True
    
    cmd = [
        sys.executable, "src/train.py",
        "--train_file", train_file,
        "--test_file", test_file,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", "1e-5",  # Оптимизировано для Vinthroy
        "--warmup_steps", "50",     # Уменьшено для Vinthroy
        "--gradient_accumulation_steps", "8"  # Оптимизировано для Vinthroy
    ]
    
    return run_command(cmd, "Обучение модели (оптимизированное)")

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
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ОБРАБОТКИ ДОКУМЕНТОВ")
    logger.info("=" * 80)
    logger.info(f"📁 Входная директория: {args.input_dir or 'data/raw'}")
    logger.info(f"🎯 Максимум документов: {args.max_docs}")
    logger.info(f"🧠 Эпохи обучения: {args.epochs}")
    logger.info(f"📦 Размер батча: {args.batch_size}")
    logger.info("=" * 80)
    
    # Проверки
    logger.info("🔍 Проверка системы...")
    if not check_dependencies():
        return False
    
    if not check_openai_key():
        return False
    
    create_directories()
    
    # Выполнение шагов
    steps = [
        ("📄 Предобработка PDF", lambda: step1_preprocess(args.input_dir)),
        ("🤖 Анализ с OpenAI", lambda: step2_analyze_with_openai(max_docs=args.max_docs, max_workers=args.max_workers)),
        ("📊 Создание датасета", step3_build_dataset),
        ("🧠 Обучение модели", lambda: step4_train_model(epochs=args.epochs, batch_size=args.batch_size)),
        ("✅ Тестирование модели", step5_test_model)
    ]
    
    completed_steps = 0
    for i, (step_name, step_func) in enumerate(steps, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 Шаг {i}/{len(steps)}: {step_name}")
        logger.info(f"{'='*60}")
        
        step_start = time.time()
        if not step_func():
            logger.error(f"❌ Пайплайн остановлен на шаге: {step_name}")
            return False
        
        step_time = time.time() - step_start
        completed_steps += 1
        logger.info(f"✅ Шаг {i} завершен за {step_time:.2f} секунд")
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
    logger.info("=" * 80)
    logger.info(f"📊 Статистика:")
    logger.info(f"   ✅ Выполнено шагов: {completed_steps}/{len(steps)}")
    logger.info(f"   ⏱️  Общее время: {total_time:.2f} секунд ({total_time/60:.1f} минут)")
    logger.info(f"   🚀 Среднее время на шаг: {total_time/len(steps):.2f} секунд")
    logger.info("=" * 80)
    logger.info("🎯 Следующие шаги:")
    logger.info("   📱 Для запуска GUI: python gui/app.py")
    logger.info("   🧪 Для тестирования: python demo.py")
    logger.info("   📊 Для мониторинга: python src/monitor_training.py")
    logger.info("=" * 80)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Полный пайплайн обработки юридических документов')
    parser.add_argument('--input-dir', type=str, help='Директория с исходными PDF файлами')
    parser.add_argument('--max-docs', type=int, default=50000, help='Максимальное количество документов для анализа')
    parser.add_argument('--max-workers', type=int, default=10, help='Количество параллельных потоков для анализа')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения (оптимизировано для больших датасетов)')
    parser.add_argument('--batch-size', type=int, default=8, help='Размер батча (оптимизировано для больших датасетов)')
    parser.add_argument('--skip-training', action='store_true', help='Пропустить обучение модели')
    
    args = parser.parse_args()
    
    if args.skip_training:
        # Только предобработка и анализ
        logger.info("Запуск без обучения модели...")
        check_dependencies()
        check_openai_key()
        create_directories()
        
        step1_preprocess(args.input_dir)
        step2_analyze_with_openai(max_docs=args.max_docs, max_workers=args.max_workers)
        step3_build_dataset()
        
        logger.info("Предобработка завершена. Для обучения запустите: python src/train.py")
    else:
        # Полный пайплайн
        run_full_pipeline(args)

if __name__ == '__main__':
    main() 