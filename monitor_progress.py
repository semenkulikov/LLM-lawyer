#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from loguru import logger

def count_files(directory, extension):
    """Подсчет файлов с определенным расширением"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(extension)])

def monitor_progress():
    """Мониторинг прогресса обработки документов"""
    logger.info("🔍 Мониторинг прогресса обработки документов")
    logger.info("=" * 60)
    
    # Подсчет файлов в разных директориях
    raw_pdfs = count_files("data/raw", ".pdf")
    processed_txts = count_files("data/processed", ".txt")
    analyzed_jsons = count_files("data/analyzed", ".json")
    
    logger.info(f"📄 PDF файлов в data/raw: {raw_pdfs}")
    logger.info(f"📝 TXT файлов в data/processed: {processed_txts}")
    logger.info(f"🤖 JSON файлов в data/analyzed: {analyzed_jsons}")
    
    if raw_pdfs > 0:
        preprocessing_progress = (processed_txts / raw_pdfs) * 100
        analysis_progress = (analyzed_jsons / raw_pdfs) * 100
        
        logger.info(f"📊 Прогресс предобработки: {preprocessing_progress:.1f}%")
        logger.info(f"📊 Прогресс анализа OpenAI: {analysis_progress:.1f}%")
        
        # Оценка оставшегося времени
        if analyzed_jsons > 0:
            remaining_docs = raw_pdfs - analyzed_jsons
            if remaining_docs > 0:
                # Примерная оценка: 1 документ в минуту через OpenAI API
                estimated_minutes = remaining_docs
                estimated_hours = estimated_minutes / 60
                
                logger.info(f"⏱️  Осталось документов: {remaining_docs}")
                logger.info(f"⏱️  Примерное время до завершения: {estimated_hours:.1f} часов")
    
    logger.info("=" * 60)

def main():
    """Основная функция"""
    while True:
        try:
            monitor_progress()
            print("\n" + "=" * 60)
            print("🔄 Обновление через 30 секунд... (Ctrl+C для выхода)")
            print("=" * 60)
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("👋 Мониторинг завершен")
            break

if __name__ == "__main__":
    main() 