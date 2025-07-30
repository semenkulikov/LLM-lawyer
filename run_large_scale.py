#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import time
import json
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

def setup_logging():
    """Настройка логирования для больших объемов"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/large_scale.log",
        rotation="100 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )

def check_system_requirements():
    """Проверка системных требований для больших объемов"""
    logger.info("🔍 Проверка системных требований...")
    
    # Проверка RAM
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb < 16:
        logger.warning(f"⚠️  Рекомендуется минимум 16GB RAM, доступно: {ram_gb:.1f}GB")
    else:
        logger.info(f"✅ RAM: {ram_gb:.1f}GB - OK")
    
    # Проверка дискового пространства
    disk_usage = psutil.disk_usage('.')
    disk_gb = disk_usage.free / (1024**3)
    if disk_gb < 50:
        logger.warning(f"⚠️  Рекомендуется минимум 50GB свободного места, доступно: {disk_gb:.1f}GB")
    else:
        logger.info(f"✅ Свободное место: {disk_gb:.1f}GB - OK")
    
    # Проверка CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️  CUDA недоступна, будет использован CPU (очень медленно для 50k+ документов)")
    except ImportError:
        logger.error("❌ PyTorch не установлен")

def count_documents(input_dir):
    """Подсчет документов в директории"""
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    return len(pdf_files)

def estimate_processing_time(num_docs):
    """Оценка времени обработки"""
    # Примерные оценки (в часах)
    preprocessing_time = num_docs * 0.001  # 3.6 секунды на документ
    analysis_time = num_docs * 0.01       # 36 секунд на документ (OpenAI API)
    training_time = num_docs * 0.0001     # 0.36 секунды на документ
    
    total_time = preprocessing_time + analysis_time + training_time
    
    logger.info(f"📊 Оценка времени обработки для {num_docs:,} документов:")
    logger.info(f"   Предобработка: {preprocessing_time:.1f} часов")
    logger.info(f"   Анализ OpenAI: {analysis_time:.1f} часов")
    logger.info(f"   Обучение: {training_time:.1f} часов")
    logger.info(f"   Общее время: {total_time:.1f} часов ({total_time/24:.1f} дней)")

def create_batch_processing_script(num_docs, batch_size=100):
    """Создание скрипта для пакетной обработки"""
    script_content = f"""#!/usr/bin/env python
# Автоматически созданный скрипт для пакетной обработки {num_docs:,} документов

import os
import sys
import subprocess
from pathlib import Path

def process_batch(start_idx, end_idx, batch_num):
    print(f"🔄 Обработка батча {batch_num}: документы {start_idx:,}-{end_idx:,}")
    
    # Предобработка батча
    cmd = [
        "python", "src/preprocess.py",
        "--input-dir", "data/raw",
        "--output-dir", f"data/processed/batch_{batch_num}",
        "--start-index", str(start_idx),
        "--end-index", str(end_idx)
    ]
    subprocess.run(cmd, check=True)
    
    # Анализ батча
    cmd = [
        "python", "src/process_with_openai.py",
        "--input-dir", f"data/processed/batch_{batch_num}",
        "--output-dir", f"data/analyzed/batch_{batch_num}",
        "--max-workers", "3"
    ]
    subprocess.run(cmd, check=True)

def main():
    total_docs = {num_docs}
    batch_size = {batch_size}
    
    for i in range(0, total_docs, batch_size):
        batch_num = i // batch_size + 1
        start_idx = i
        end_idx = min(i + batch_size, total_docs)
        
        try:
            process_batch(start_idx, end_idx, batch_num)
            print(f"✅ Батч {batch_num} завершен успешно")
        except Exception as e:
            print(f"❌ Ошибка в батче {batch_num}: {{e}}")
            break

if __name__ == "__main__":
    main()
"""
    
    with open("process_batches.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    logger.info("📝 Создан скрипт process_batches.py для пакетной обработки")

def run_large_scale_pipeline(args):
    """Запуск пайплайна для больших объемов данных"""
    logger.info("🚀 Запуск пайплайна для больших объемов данных")
    
    # Проверка входных данных
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"❌ Директория {input_dir} не существует")
        return False
    
    num_docs = count_documents(input_dir)
    logger.info(f"📄 Найдено документов: {num_docs:,}")
    
    if num_docs == 0:
        logger.error("❌ Документы не найдены")
        return False
    
    # Оценка времени
    estimate_processing_time(num_docs)
    
    # Создание пакетного скрипта
    create_batch_processing_script(num_docs, args.batch_size)
    
    # Запуск обработки
    if args.mode == "batch":
        logger.info("🔄 Запуск пакетной обработки...")
        cmd = ["python", "process_batches.py"]
        subprocess.run(cmd, check=True)
    else:
        logger.info("🔄 Запуск полного пайплайна...")
        cmd = [
            "python", "run_pipeline.py",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--max-docs", str(num_docs)
        ]
        subprocess.run(cmd, check=True)
    
    logger.info("✅ Обработка завершена")
    return True

def main():
    parser = argparse.ArgumentParser(description='Пайплайн для обработки 50,000+ документов')
    parser.add_argument('--input-dir', type=str, default='data/raw', help='Директория с PDF документами')
    parser.add_argument('--batch-size', type=int, default=1000, help='Размер батча для обработки')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')
    parser.add_argument('--mode', choices=['batch', 'full'], default='batch', help='Режим обработки')
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging()
    
    # Проверка требований
    check_system_requirements()
    
    # Запуск пайплайна
    success = run_large_scale_pipeline(args)
    
    if success:
        logger.info("🎉 Обработка завершена успешно!")
    else:
        logger.error("❌ Обработка завершена с ошибками")
        sys.exit(1)

if __name__ == "__main__":
    main() 