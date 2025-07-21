#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import sys
from loguru import logger

def check_cuda():
    """Проверка доступности CUDA"""
    logger.info("🔍 Диагностика CUDA")
    logger.info("=" * 50)
    
    # Проверка PyTorch
    try:
        import torch
        logger.info(f"PyTorch версия: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch не установлен")
        return False
    
    # Проверка CUDA
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA доступна: {cuda_available}")
    
    if cuda_available:
        logger.success("✅ CUDA работает!")
        logger.info(f"CUDA версия: {torch.version.cuda}")
        logger.info(f"Количество GPU: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Тест CUDA
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            logger.success("✅ CUDA тест пройден успешно!")
        except Exception as e:
            logger.error(f"❌ Ошибка CUDA теста: {e}")
            return False
    else:
        logger.error("❌ CUDA недоступна!")
        logger.info("Возможные причины:")
        logger.info("1. Не установлен CUDA toolkit")
        logger.info("2. PyTorch установлен без поддержки CUDA")
        logger.info("3. Нет совместимой видеокарты NVIDIA")
        logger.info("4. Драйверы NVIDIA не установлены")
        
        # Проверка установки PyTorch
        logger.info("\nПроверка установки PyTorch:")
        try:
            import torch
            logger.info(f"PyTorch установлен: {torch.__version__}")
            
            # Проверка, есть ли CUDA в PyTorch
            if hasattr(torch, 'cuda'):
                logger.info("PyTorch имеет модуль CUDA")
            else:
                logger.error("PyTorch не имеет модуля CUDA")
                
        except ImportError:
            logger.error("PyTorch не установлен")
            return False
    
    return cuda_available

def check_system():
    """Проверка системы"""
    logger.info("\n🔧 Информация о системе")
    logger.info("=" * 50)
    
    import platform
    logger.info(f"ОС: {platform.system()} {platform.release()}")
    logger.info(f"Архитектура: {platform.machine()}")
    logger.info(f"Python: {sys.version}")
    
    # Проверка переменных окружения
    import os
    cuda_home = os.environ.get('CUDA_HOME')
    cuda_path = os.environ.get('CUDA_PATH')
    
    if cuda_home:
        logger.info(f"CUDA_HOME: {cuda_home}")
    else:
        logger.warning("CUDA_HOME не установлена")
    
    if cuda_path:
        logger.info(f"CUDA_PATH: {cuda_path}")
    else:
        logger.warning("CUDA_PATH не установлена")

def main():
    """Основная функция"""
    logger.info("🚀 Диагностика системы для PyTorch + CUDA")
    logger.info("=" * 60)
    
    check_system()
    cuda_works = check_cuda()
    
    logger.info("\n" + "=" * 60)
    if cuda_works:
        logger.success("🎉 CUDA работает корректно! Модель будет обучаться на GPU.")
        logger.info("Рекомендации:")
        logger.info("- Увеличьте batch_size для ускорения обучения")
        logger.info("- Используйте fp16=True для экономии памяти")
    else:
        logger.warning("⚠️ CUDA недоступна. Модель будет обучаться на CPU.")
        logger.info("Рекомендации:")
        logger.info("- Установите CUDA toolkit и PyTorch с поддержкой CUDA")
        logger.info("- Уменьшите batch_size для экономии памяти CPU")
        logger.info("- Обучение будет медленнее, но все равно возможно")

if __name__ == '__main__':
    main() 