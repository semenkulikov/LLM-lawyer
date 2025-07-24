#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

def setup_logging():
    """Настройка логирования"""
    logger.remove()
    logger.add(
        "logs/model_manager.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(lambda msg: print(msg, end=""), level="INFO")

class ModelManager:
    """Менеджер для управления версиями моделей"""
    
    def __init__(self, models_dir="models", max_versions=10):
        self.models_dir = Path(models_dir)
        self.max_versions = max_versions
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def list_models(self):
        """Список всех доступных моделей"""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_info = self.get_model_info(model_dir)
                if model_info:
                    models.append(model_info)
        
        # Сортируем по дате создания
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models
    
    def get_model_info(self, model_path):
        """Получение информации о модели"""
        try:
            # Проверяем наличие основных файлов модели
            required_files = ['config.json', 'tokenizer.json']
            # Проверяем наличие модели (pytorch_model.bin или model.safetensors)
            model_files = ['pytorch_model.bin', 'model.safetensors']
            has_model_file = any((model_path / file).exists() for file in model_files)
            
            if not all((model_path / file).exists() for file in required_files) or not has_model_file:
                return None
            
            # Получаем информацию о размере
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            
            # Получаем дату создания
            created_at = datetime.fromtimestamp(model_path.stat().st_ctime)
            
            # Загружаем дополнительную информацию
            info_file = model_path / "model_info.json"
            additional_info = {}
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    additional_info = json.load(f)
            
            return {
                'name': model_path.name,
                'path': str(model_path),
                'size_mb': round(total_size / (1024 * 1024), 2),
                'created_at': created_at,
                'files_count': len(list(model_path.rglob('*'))),
                **additional_info
            }
            
        except Exception as e:
            logger.error(f"Ошибка при получении информации о модели {model_path}: {str(e)}")
            return None
    
    def create_backup(self, model_path, backup_name=None):
        """Создание резервной копии модели"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Модель не найдена: {model_path}")
            return None
        
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{model_path.name}_{timestamp}"
        
        backup_path = self.models_dir / backup_name
        
        try:
            # Копируем модель
            shutil.copytree(model_path, backup_path)
            
            # Добавляем информацию о резервной копии
            backup_info = {
                "backup_date": datetime.now().isoformat(),
                "original_model": str(model_path),
                "backup_type": "manual"
            }
            
            with open(backup_path / "backup_info.json", 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Резервная копия создана: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Ошибка при создании резервной копии: {str(e)}")
            return None
    
    def restore_model(self, backup_path, target_name=None):
        """Восстановление модели из резервной копии"""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"Резервная копия не найдена: {backup_path}")
            return None
        
        if target_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_name = f"restored_{backup_path.name}_{timestamp}"
        
        target_path = self.models_dir / target_name
        
        try:
            # Копируем модель
            shutil.copytree(backup_path, target_path)
            
            # Добавляем информацию о восстановлении
            restore_info = {
                "restore_date": datetime.now().isoformat(),
                "backup_source": str(backup_path),
                "restore_type": "manual"
            }
            
            with open(target_path / "restore_info.json", 'w', encoding='utf-8') as f:
                json.dump(restore_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Модель восстановлена: {target_path}")
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Ошибка при восстановлении модели: {str(e)}")
            return None
    
    def cleanup_old_models(self, keep_count=None):
        """Очистка старых версий моделей"""
        if keep_count is None:
            keep_count = self.max_versions
        
        models = self.list_models()
        
        if len(models) <= keep_count:
            logger.info(f"Количество моделей ({len(models)}) не превышает лимит ({keep_count})")
            return
        
        # Удаляем старые модели
        models_to_remove = models[keep_count:]
        
        for model in models_to_remove:
            try:
                model_path = Path(model['path'])
                if model_path.exists():
                    shutil.rmtree(model_path)
                    logger.info(f"Удалена старая модель: {model['name']}")
            except Exception as e:
                logger.error(f"Ошибка при удалении модели {model['name']}: {str(e)}")
        
        logger.info(f"Удалено {len(models_to_remove)} старых моделей")
    
    def compare_models(self, model1_path, model2_path):
        """Сравнение двух моделей"""
        model1_info = self.get_model_info(Path(model1_path))
        model2_info = self.get_model_info(Path(model2_path))
        
        if not model1_info or not model2_info:
            logger.error("Не удалось получить информацию об одной из моделей")
            return None
        
        comparison = {
            'model1': model1_info,
            'model2': model2_info,
            'differences': {}
        }
        
        # Сравниваем размеры
        size_diff = model1_info['size_mb'] - model2_info['size_mb']
        comparison['differences']['size_mb'] = {
            'model1': model1_info['size_mb'],
            'model2': model2_info['size_mb'],
            'difference': round(size_diff, 2)
        }
        
        # Сравниваем даты создания
        date_diff = model1_info['created_at'] - model2_info['created_at']
        comparison['differences']['age_days'] = {
            'model1': model1_info['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
            'model2': model2_info['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
            'difference_days': date_diff.days
        }
        
        # Сравниваем количество файлов
        files_diff = model1_info['files_count'] - model2_info['files_count']
        comparison['differences']['files_count'] = {
            'model1': model1_info['files_count'],
            'model2': model2_info['files_count'],
            'difference': files_diff
        }
        
        return comparison
    
    def export_model(self, model_path, export_dir):
        """Экспорт модели в указанную директорию"""
        model_path = Path(model_path)
        export_dir = Path(export_dir)
        
        if not model_path.exists():
            logger.error(f"Модель не найдена: {model_path}")
            return None
        
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Копируем модель
            target_path = export_dir / model_path.name
            shutil.copytree(model_path, target_path)
            
            # Добавляем информацию об экспорте
            export_info = {
                "export_date": datetime.now().isoformat(),
                "source_model": str(model_path),
                "export_type": "manual"
            }
            
            with open(target_path / "export_info.json", 'w', encoding='utf-8') as f:
                json.dump(export_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Модель экспортирована: {target_path}")
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте модели: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Управление версиями моделей')
    parser.add_argument('--models_dir', type=str, default='models', help='Директория с моделями')
    parser.add_argument('--max_versions', type=int, default=10, help='Максимальное количество версий')
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда list
    list_parser = subparsers.add_parser('list', help='Список всех моделей')
    
    # Команда backup
    backup_parser = subparsers.add_parser('backup', help='Создание резервной копии')
    backup_parser.add_argument('model_path', type=str, help='Путь к модели')
    backup_parser.add_argument('--name', type=str, help='Имя резервной копии')
    
    # Команда restore
    restore_parser = subparsers.add_parser('restore', help='Восстановление модели')
    restore_parser.add_argument('backup_path', type=str, help='Путь к резервной копии')
    restore_parser.add_argument('--name', type=str, help='Имя восстановленной модели')
    
    # Команда cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Очистка старых моделей')
    cleanup_parser.add_argument('--keep', type=int, help='Количество моделей для сохранения')
    
    # Команда compare
    compare_parser = subparsers.add_parser('compare', help='Сравнение моделей')
    compare_parser.add_argument('model1', type=str, help='Путь к первой модели')
    compare_parser.add_argument('model2', type=str, help='Путь ко второй модели')
    
    # Команда export
    export_parser = subparsers.add_parser('export', help='Экспорт модели')
    export_parser.add_argument('model_path', type=str, help='Путь к модели')
    export_parser.add_argument('export_dir', type=str, help='Директория для экспорта')
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging()
    
    # Создание менеджера моделей
    manager = ModelManager(args.models_dir, args.max_versions)
    
    if args.command == 'list':
        models = manager.list_models()
        if models:
            print(f"\nНайдено {len(models)} моделей:\n")
            for model in models:
                print(f"📁 {model['name']}")
                print(f"   Путь: {model['path']}")
                print(f"   Размер: {model['size_mb']} MB")
                print(f"   Создана: {model['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Файлов: {model['files_count']}")
                print()
        else:
            print("Модели не найдены")
    
    elif args.command == 'backup':
        backup_path = manager.create_backup(args.model_path, args.name)
        if backup_path:
            print(f"✅ Резервная копия создана: {backup_path}")
        else:
            print("❌ Ошибка при создании резервной копии")
    
    elif args.command == 'restore':
        restored_path = manager.restore_model(args.backup_path, args.name)
        if restored_path:
            print(f"✅ Модель восстановлена: {restored_path}")
        else:
            print("❌ Ошибка при восстановлении модели")
    
    elif args.command == 'cleanup':
        keep_count = args.keep if args.keep else manager.max_versions
        manager.cleanup_old_models(keep_count)
        print(f"✅ Очистка завершена. Сохранено {keep_count} моделей")
    
    elif args.command == 'compare':
        comparison = manager.compare_models(args.model1, args.model2)
        if comparison:
            print("\n📊 Сравнение моделей:\n")
            for key, diff in comparison['differences'].items():
                print(f"{key}:")
                for k, v in diff.items():
                    print(f"  {k}: {v}")
                print()
        else:
            print("❌ Ошибка при сравнении моделей")
    
    elif args.command == 'export':
        export_path = manager.export_model(args.model_path, args.export_dir)
        if export_path:
            print(f"✅ Модель экспортирована: {export_path}")
        else:
            print("❌ Ошибка при экспорте модели")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 