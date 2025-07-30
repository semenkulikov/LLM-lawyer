#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
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
                    os.environ[key.strip()] = value.strip()
        logger.success("✅ Переменные окружения загружены")
    else:
        logger.warning("⚠️  Файл .env не найден")

# Загружаем переменные окружения при запуске
load_env()

def test_dependencies():
    """Тестирование зависимостей"""
    logger.info("🔍 Проверка зависимостей...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'openai', 
        'pymupdf', 'nltk', 'tqdm', 'numpy', 'loguru'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.success(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error(f"Отсутствуют пакеты: {', '.join(missing_packages)}")
        logger.info("Установите их командой: pip install -r requirements.txt")
        return False
    
    logger.success("✅ Все зависимости установлены")
    return True

def test_openai_key():
    """Тестирование OpenAI API ключа"""
    logger.info("🔑 Проверка OpenAI API ключа...")
    
    # Сначала проверяем .env файл
    try:
        result = subprocess.run([
            sys.executable, "check_env.py"
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
        
        logger.success("✅ .env файл настроен правильно")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Проблемы с .env файлом: {e.stderr}")
        logger.info("Запустите check_env.bat для диагностики")
        return False
    
    # Затем тестируем API
    try:
        result = subprocess.run([
            sys.executable, "src/test_openai_key.py"
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
        
        logger.success("✅ OpenAI API ключ работает")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ошибка OpenAI API: {e.stderr}")
        return False

def test_directory_structure():
    """Проверка структуры директорий"""
    logger.info("📁 Проверка структуры директорий...")
    
    required_dirs = [
        'src', 'data', 'data/raw', 'data/processed', 
        'data/analyzed', 'models', 'gui'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if Path(directory).exists():
            logger.success(f"✅ {directory}")
        else:
            missing_dirs.append(directory)
            logger.warning(f"⚠️  {directory} (будет создана)")
    
    # Создаем недостающие директории
    for directory in missing_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Создана директория: {directory}")
    
    return True

def test_scripts():
    """Проверка доступности скриптов"""
    logger.info("📜 Проверка скриптов...")
    
    required_scripts = [
        'src/preprocess.py',
        'src/process_with_openai.py', 
        'src/build_dataset.py',
        'src/train.py',
        'src/inference.py',
        'src/test_example.py',
        'src/monitor_training.py',
        'gui/app.py',
        'run_pipeline.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if Path(script).exists():
            logger.success(f"✅ {script}")
        else:
            missing_scripts.append(script)
            logger.error(f"❌ {script}")
    
    if missing_scripts:
        logger.error(f"Отсутствуют скрипты: {', '.join(missing_scripts)}")
        return False
    
    logger.success("✅ Все скрипты найдены")
    return True

def test_sample_data():
    """Проверка наличия тестовых данных"""
    logger.info("📄 Проверка тестовых данных...")
    
    raw_files = list(Path('data/raw').glob('*.pdf'))
    processed_files = list(Path('data/processed').glob('*.txt'))
    analyzed_files = list(Path('data/analyzed').glob('*_analyzed.json'))
    
    logger.info(f"📊 PDF файлов в data/raw: {len(raw_files)}")
    logger.info(f"📊 TXT файлов в data/processed: {len(processed_files)}")
    logger.info(f"📊 JSON файлов в data/analyzed: {len(analyzed_files)}")
    
    if not raw_files:
        logger.warning("⚠️  Нет PDF файлов в data/raw/")
        logger.info("Поместите PDF документы в папку data/raw/ для тестирования")
    
    return True

def main():
    """Основная функция тестирования"""
    logger.info("🚀 Быстрое тестирование системы")
    logger.info("=" * 50)
    
    tests = [
        ("Зависимости", test_dependencies),
        ("Структура директорий", test_directory_structure),
        ("Скрипты", test_scripts),
        ("Тестовые данные", test_sample_data),
        ("OpenAI API", test_openai_key)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Ошибка в тесте {test_name}: {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    logger.info("\n" + "=" * 50)
    logger.info("📋 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            logger.success(f"✅ {test_name}: ПРОЙДЕН")
            passed += 1
        else:
            logger.error(f"❌ {test_name}: НЕ ПРОЙДЕН")
    
    logger.info(f"\n📊 Результат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        logger.success("🎉 Все тесты пройдены! Система готова к работе.")
        logger.info("\nСледующие шаги:")
        logger.info("1. Поместите PDF документы в data/raw/")
        logger.info("2. Запустите: python run_pipeline.py")
        logger.info("3. Или запустите GUI: python gui/app.py")
    else:
        logger.error("💥 Некоторые тесты не пройдены. Исправьте ошибки и повторите.")
        logger.info("\nРекомендации:")
        logger.info("1. Установите недостающие зависимости: pip install -r requirements.txt")
        logger.info("2. Проверьте OpenAI API ключ: python src/test_openai_key.py")
        logger.info("3. Убедитесь, что все файлы проекта на месте")

if __name__ == '__main__':
    main()