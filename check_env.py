#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from loguru import logger

def check_env_file():
    """Проверка .env файла"""
    logger.info("🔍 Проверка .env файла...")
    
    env_file = Path('.env')
    
    if not env_file.exists():
        logger.error("❌ Файл .env не найден!")
        logger.info("Создайте его командой: copy env.example .env")
        return False
    
    logger.success(f"✅ Файл .env найден: {env_file.absolute()}")
    
    # Читаем и проверяем содержимое
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info("📄 Содержимое .env файла:")
        print("-" * 50)
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                if line.startswith('#'):
                    logger.info(f"  {i:2d}: {line}")
                elif '=' in line:
                    key, value = line.split('=', 1)
                    if 'API_KEY' in key and 'your-' in value:
                        logger.warning(f"  {i:2d}: {key}={value[:20]}... (НЕ НАСТРОЕНО!)")
                    else:
                        logger.success(f"  {i:2d}: {key}={value[:20]}...")
                else:
                    logger.info(f"  {i:2d}: {line}")
        
        print("-" * 50)
        
        # Проверяем конкретные переменные
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("❌ OPENAI_API_KEY не найден в переменных окружения")
            logger.info("Загружаем из .env файла...")
            
            # Загружаем вручную
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Убираем кавычки из значения
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
            
            api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            if api_key.startswith('sk-') and 'your-' not in api_key:
                logger.success(f"✅ OPENAI_API_KEY найден: {api_key[:10]}...")
                return True
            else:
                logger.error("❌ OPENAI_API_KEY не настроен правильно")
                logger.info("Замените 'sk-your-openai-api-key-here' на ваш реальный ключ")
                return False
        else:
            logger.error("❌ OPENAI_API_KEY не найден")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка чтения .env файла: {e}")
        return False

def main():
    """Основная функция"""
    logger.info("🚀 Проверка конфигурации .env")
    logger.info("=" * 50)
    
    success = check_env_file()
    
    logger.info("=" * 50)
    if success:
        logger.success("🎉 .env файл настроен правильно!")
    else:
        logger.error("💥 Проблемы с .env файлом. Исправьте и повторите.")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 