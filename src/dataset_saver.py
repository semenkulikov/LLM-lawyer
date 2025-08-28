#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

class HybridDatasetSaver:
    """Сохранение результатов гибридной обработки для будущего обучения"""
    
    def __init__(self, output_dir="datasets/hybrid_generated"):
        """
        Инициализация сохранения датасета
        
        Args:
            output_dir: Папка для сохранения датасета
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем файл с временной меткой
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_path = self.output_dir / f"dataset_{timestamp}.jsonl"
        
        # Открываем файл для записи
        self.file = open(self.file_path, "a", encoding="utf-8")
        
        logger.info(f"Датасет будет сохраняться в: {self.file_path}")
    
    def save_example(self, facts: str, local_response: str, hybrid_response: str, 
                    provider: str = "openai", mode: str = "polish"):
        """
        Сохранение примера в датасет
        
        Args:
            facts: Исходные фактические обстоятельства
            local_response: Ответ локальной модели
            hybrid_response: Ответ после обработки внешним LLM
            provider: Использованный провайдер (openai/gemini)
            mode: Режим обработки (polish/enhance/verify)
        """
        try:
            record = {
                "instruction": "Составь исковое заявление по фактическим обстоятельствам дела.",
                "input": facts.strip(),
                "output": hybrid_response.strip(),
                "metadata": {
                    "local_response": local_response.strip(),
                    "provider": provider,
                    "mode": mode,
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            
            # Записываем в JSONL формат
            self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.file.flush()  # Принудительная запись на диск
            
            logger.info(f"Сохранен пример в датасет: {len(facts)} символов -> {len(hybrid_response)} символов")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения примера: {e}")
    
    def get_stats(self):
        """Получение статистики датасета"""
        try:
            count = 0
            total_input_chars = 0
            total_output_chars = 0
            
            # Читаем все записи для подсчета статистики
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        count += 1
                        total_input_chars += len(record.get('input', ''))
                        total_output_chars += len(record.get('output', ''))
            
            return {
                'total_examples': count,
                'total_input_chars': total_input_chars,
                'total_output_chars': total_output_chars,
                'avg_input_length': total_input_chars // count if count > 0 else 0,
                'avg_output_length': total_output_chars // count if count > 0 else 0,
                'file_path': str(self.file_path)
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    def close(self):
        """Закрытие файла"""
        if self.file:
            self.file.close()
            logger.info(f"Датасет сохранен: {self.file_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def create_dataset_saver(output_dir="datasets/hybrid_generated"):
    """Фабричная функция для создания сохранения датасета"""
    return HybridDatasetSaver(output_dir)
