#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import schedule
from pathlib import Path
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import threading
from loguru import logger

class NewDataHandler(FileSystemEventHandler):
    """
    Обработчик новых файлов данных для автоматического дообучения
    """
    
    def __init__(self, model_path, output_dir, config):
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config
        self.processed_files = set()
        self.lock = threading.Lock()
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.jsonl'):
            self.handle_new_data(event.src_path)
    
    def on_moved(self, event):
        if not event.is_directory and event.src_path.endswith('.jsonl'):
            self.handle_new_data(event.src_path)
    
    def handle_new_data(self, file_path):
        """Обработка нового файла данных"""
        with self.lock:
            if file_path in self.processed_files:
                return
                
            logger.info(f"Обнаружен новый файл данных: {file_path}")
            
            # Проверяем размер файла и время последней модификации
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            
            # Ждем, пока файл перестанет изменяться
            time.sleep(5)
            
            # Проверяем, что файл не пустой
            if file_size == 0:
                logger.warning(f"Файл {file_path} пустой, пропускаем")
                return
            
            # Проверяем количество примеров в файле
            example_count = self.count_examples(file_path)
            if example_count < self.config.get('min_examples', 10):
                logger.warning(f"Файл {file_path} содержит мало примеров ({example_count}), пропускаем")
                return
            
            # Запускаем дообучение
            success = self.run_incremental_training(file_path)
            
            if success:
                self.processed_files.add(file_path)
                logger.info(f"Файл {file_path} успешно обработан")
            else:
                logger.error(f"Ошибка при обработке файла {file_path}")
    
    def count_examples(self, file_path):
        """Подсчет количества примеров в файле"""
        try:
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        except Exception as e:
            logger.error(f"Ошибка при подсчете примеров: {str(e)}")
            return 0
    
    def run_incremental_training(self, data_file):
        """Запуск инкрементального дообучения"""
        try:
            # Создаем уникальную директорию для этой итерации
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            iteration_dir = Path(self.output_dir) / f"incremental_{timestamp}"
            
            # Команда для запуска инкрементального обучения
            cmd = [
                "python", "src/incremental_train.py",
                "--model_path", self.model_path,
                "--new_data", data_file,
                "--output_dir", str(iteration_dir),
                "--epochs", str(self.config.get('epochs', 3)),
                "--batch_size", str(self.config.get('batch_size', 4)),
                "--learning_rate", str(self.config.get('learning_rate', 5e-5)),
                "--create_backup"
            ]
            
            logger.info(f"Запуск команды: {' '.join(cmd)}")
            
            # Запускаем процесс
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get('timeout', 3600)  # 1 час таймаут
            )
            
            if result.returncode == 0:
                logger.info("Инкрементальное обучение завершено успешно")
                
                # Обновляем путь к модели на новую версию
                self.model_path = str(iteration_dir)
                
                # Сохраняем информацию о дообучении
                self.save_training_info(data_file, iteration_dir, result.stdout)
                
                return True
            else:
                logger.error(f"Ошибка при обучении: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Таймаут при обучении")
            return False
        except Exception as e:
            logger.error(f"Ошибка при запуске обучения: {str(e)}")
            return False
    
    def save_training_info(self, data_file, iteration_dir, output):
        """Сохранение информации о дообучении"""
        info = {
            "data_file": data_file,
            "iteration_dir": str(iteration_dir),
            "timestamp": datetime.now().isoformat(),
            "output": output
        }
        
        info_file = Path(self.output_dir) / "auto_training_history.json"
        
        # Загружаем существующую историю
        history = []
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # Добавляем новую запись
        history.append(info)
        
        # Сохраняем обновленную историю
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

def setup_logging():
    """Настройка логирования"""
    logger.remove()
    logger.add(
        "logs/auto_incremental.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(lambda msg: print(msg, end=""), level="INFO")

def load_config(config_file):
    """Загрузка конфигурации"""
    if not os.path.exists(config_file):
        # Создаем конфигурацию по умолчанию
        default_config = {
            "watch_directory": "data/new",
            "model_path": "models/legal_model",
            "output_dir": "models/incremental",
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "min_examples": 10,
            "timeout": 3600,
            "schedule_enabled": False,
            "schedule_time": "02:00",
            "backup_enabled": True
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Создана конфигурация по умолчанию: {config_file}")
        return default_config
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def scheduled_training(config, model_path, output_dir):
    """Планируемое дообучение"""
    logger.info("Запуск планируемого дообучения")
    
    # Ищем новые файлы данных
    watch_dir = Path(config['watch_directory'])
    if not watch_dir.exists():
        logger.warning(f"Директория для мониторинга не существует: {watch_dir}")
        return
    
    new_files = list(watch_dir.glob("*.jsonl"))
    if not new_files:
        logger.info("Новых файлов данных не найдено")
        return
    
    # Обрабатываем все новые файлы
    handler = NewDataHandler(model_path, output_dir, config)
    
    for file_path in new_files:
        logger.info(f"Обработка файла: {file_path}")
        success = handler.run_incremental_training(str(file_path))
        
        if success:
            # Перемещаем обработанный файл в архив
            archive_dir = watch_dir / "processed"
            archive_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = archive_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            file_path.rename(archive_path)
            logger.info(f"Файл перемещен в архив: {archive_path}")

def main():
    parser = argparse.ArgumentParser(description='Автоматическое инкрементальное дообучение модели')
    parser.add_argument('--config', type=str, default='config/auto_incremental.json',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--watch', action='store_true', help='Запустить мониторинг файлов')
    parser.add_argument('--schedule', action='store_true', help='Запустить планировщик')
    parser.add_argument('--run-now', action='store_true', help='Запустить обработку немедленно')
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Создание необходимых директорий
    Path(config['watch_directory']).mkdir(parents=True, exist_ok=True)
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    logger.info("Автоматическое инкрементальное дообучение запущено")
    logger.info(f"Конфигурация: {config}")
    
    if args.run_now:
        # Немедленный запуск
        scheduled_training(config, config['model_path'], config['output_dir'])
        return
    
    if args.watch:
        # Запуск мониторинга файлов
        event_handler = NewDataHandler(config['model_path'], config['output_dir'], config)
        observer = Observer()
        observer.schedule(event_handler, config['watch_directory'], recursive=False)
        observer.start()
        
        logger.info(f"Мониторинг запущен для директории: {config['watch_directory']}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("Мониторинг остановлен")
        
        observer.join()
    
    if args.schedule:
        # Запуск планировщика
        schedule_time = config.get('schedule_time', '02:00')
        schedule.every().day.at(schedule_time).do(
            scheduled_training, config, config['model_path'], config['output_dir']
        )
        
        logger.info(f"Планировщик запущен. Время выполнения: {schedule_time}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Проверяем каждую минуту
        except KeyboardInterrupt:
            logger.info("Планировщик остановлен")
    
    if not args.watch and not args.schedule and not args.run_now:
        logger.info("Используйте --watch для мониторинга файлов, --schedule для планировщика или --run-now для немедленного запуска")

if __name__ == '__main__':
    main() 