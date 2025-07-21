#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
from pathlib import Path
from loguru import logger

def clean_directory(directory_path, description):
    """
    Очистка директории с логированием
    
    Args:
        directory_path: Путь к директории
        description: Описание директории для логов
    """
    if os.path.exists(directory_path):
        try:
            # Подсчитываем количество файлов перед удалением
            file_count = len(list(Path(directory_path).rglob('*')))
            
            # Удаляем директорию
            shutil.rmtree(directory_path)
            logger.success(f"✅ {description}: удалено {file_count} файлов/папок")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка при очистке {description}: {e}")
            return False
    else:
        logger.info(f"ℹ️ {description}: директория не существует")
        return True

def clean_file(file_path, description):
    """
    Удаление файла с логированием
    
    Args:
        file_path: Путь к файлу
        description: Описание файла для логов
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.success(f"✅ {description}: файл удален")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка при удалении {description}: {e}")
            return False
    else:
        logger.info(f"ℹ️ {description}: файл не существует")
        return True

def clean_processed_data():
    """Очистка обработанных данных"""
    logger.info("🧹 Очистка обработанных данных...")
    
    items_to_clean = [
        ("data/structured", "Структурированные данные"),
        ("data/analyzed", "Проанализированные JSON файлы"),
        ("data/train_dataset.jsonl", "Обучающий датасет"),
        ("data/train_dataset_test.jsonl", "Тестовый датасет"),
        ("data/train_dataset_meta.json", "Метаданные датасета"),
    ]
    
    success_count = 0
    for path, description in items_to_clean:
        if path.endswith('.jsonl') or path.endswith('.json'):
            if clean_file(path, description):
                success_count += 1
        else:
            if clean_directory(path, description):
                success_count += 1
    
    return success_count == len(items_to_clean)

def clean_model_data():
    """Очистка данных модели"""
    logger.info("🧹 Очистка данных модели...")
    
    items_to_clean = [
        ("models/legal_model", "Обученная модель"),
        ("models/legal_model/logs", "Логи обучения"),
    ]
    
    success_count = 0
    for path, description in items_to_clean:
        if clean_directory(path, description):
            success_count += 1
    
    return success_count == len(items_to_clean)

def clean_results():
    """Очистка результатов"""
    logger.info("🧹 Очистка результатов...")
    
    items_to_clean = [
        ("results", "Результаты тестирования"),
    ]
    
    success_count = 0
    for path, description in items_to_clean:
        if clean_directory(path, description):
            success_count += 1
    
    return success_count == len(items_to_clean)

def clean_all():
    """Очистка всех промежуточных данных"""
    logger.info("🧹 Очистка всех промежуточных данных...")
    
    results = []
    results.append(("Обработанные данные", clean_processed_data()))
    results.append(("Данные модели", clean_model_data()))
    results.append(("Результаты", clean_results()))
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    logger.info(f"\n📊 Результат очистки: {success_count}/{total_count} успешно")
    
    for description, success in results:
        if success:
            logger.success(f"✅ {description}: очищено")
        else:
            logger.error(f"❌ {description}: ошибка при очистке")
    
    return success_count == total_count

def show_status():
    """Показать статус директорий"""
    logger.info("📊 Статус директорий проекта")
    logger.info("=" * 50)
    
    directories = [
        ("data/raw", "Исходные PDF файлы"),
        ("data/processed", "Обработанные текстовые файлы"),
        ("data/structured", "Структурированные данные"),
        ("data/analyzed", "Проанализированные JSON файлы"),
        ("models/legal_model", "Обученная модель"),
        ("results", "Результаты тестирования"),
    ]
    
    files = [
        ("data/train_dataset.jsonl", "Обучающий датасет"),
        ("data/train_dataset_test.jsonl", "Тестовый датасет"),
        ("data/train_dataset_meta.json", "Метаданные датасета"),
    ]
    
    for path, description in directories:
        if os.path.exists(path):
            file_count = len(list(Path(path).rglob('*')))
            logger.info(f"📁 {description}: {file_count} файлов/папок")
        else:
            logger.info(f"📁 {description}: не существует")
    
    for path, description in files:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            logger.info(f"📄 {description}: {size:.1f} KB")
        else:
            logger.info(f"📄 {description}: не существует")

def main():
    parser = argparse.ArgumentParser(description='Очистка промежуточных данных проекта')
    parser.add_argument('--processed', action='store_true', help='Очистить обработанные данные')
    parser.add_argument('--model', action='store_true', help='Очистить данные модели')
    parser.add_argument('--results', action='store_true', help='Очистить результаты')
    parser.add_argument('--all', action='store_true', help='Очистить все промежуточные данные')
    parser.add_argument('--status', action='store_true', help='Показать статус директорий')
    parser.add_argument('--force', action='store_true', help='Не запрашивать подтверждение')
    
    args = parser.parse_args()
    
    # Если не указаны аргументы, показываем статус
    if not any([args.processed, args.model, args.results, args.all, args.status]):
        show_status()
        return
    
    if args.status:
        show_status()
        return
    
    # Предупреждение о удалении
    if not args.force:
        logger.warning("⚠️ ВНИМАНИЕ: Это действие удалит данные безвозвратно!")
        logger.warning("Исходные PDF файлы в data/raw/ НЕ будут затронуты.")
        
        if args.all:
            logger.warning("Будут удалены: обработанные данные, модель, результаты")
        elif args.processed:
            logger.warning("Будут удалены: структурированные данные")
        elif args.model:
            logger.warning("Будут удалены: модель и логи обучения")
        elif args.results:
            logger.warning("Будут удалены: результаты тестирования")
        
        response = input("\nПродолжить? (y/N): ").strip().lower()
        if response not in ['y', 'yes', 'да']:
            logger.info("❌ Операция отменена")
            return
    
    logger.info("🚀 Начало очистки данных")
    logger.info("=" * 50)
    
    success = False
    if args.all:
        success = clean_all()
    elif args.processed:
        success = clean_processed_data()
    elif args.model:
        success = clean_model_data()
    elif args.results:
        success = clean_results()
    
    logger.info("=" * 50)
    if success:
        logger.success("🎉 Очистка завершена успешно!")
        logger.info("Теперь можно запустить пайплайн заново:")
        logger.info("python run_pipeline.py --max-docs 3 --epochs 3")
    else:
        logger.error("💥 Очистка завершена с ошибками")

if __name__ == '__main__':
    main()