#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
import os
from loguru import logger

def run_distributed_training():
    """Запуск распределенного обучения QVikhr-3-4B"""
    
    # Проверяем количество доступных GPU
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        logger.info(f"Найдено GPU: {gpu_count}")
        
        if gpu_count < 2:
            logger.warning("Для распределенного обучения требуется минимум 2 GPU")
            logger.info("Запускаем обычное обучение на одном GPU...")
            return run_single_gpu_training()
            
    except ImportError:
        logger.error("PyTorch не установлен")
        return
    
    # Параметры для распределенного обучения
    train_file = "data/train_dataset.jsonl"
    test_file = "data/train_dataset_test.jsonl"
    output_dir = "models/legal_model_distributed"
    
    # Команда для запуска распределенного обучения
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpu_count}",
        "--master_port=29500",
        "src/train_distributed.py",
        f"--train_file={train_file}",
        f"--test_file={test_file}",
        f"--output_dir={output_dir}",
        "--epochs=3",
        "--batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-5",
        "--max_length=2048"
    ]
    
    logger.info(f"Запуск распределенного обучения на {gpu_count} GPU...")
    logger.info(f"Команда: {' '.join(cmd)}")
    
    try:
        # Запускаем процесс
        process = subprocess.run(cmd, check=True)
        logger.info("Распределенное обучение завершено успешно!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при запуске распределенного обучения: {e}")
        logger.info("Попробуйте запустить обычное обучение на одном GPU")
        return run_single_gpu_training()
    
    except FileNotFoundError:
        logger.error("torchrun не найден. Убедитесь, что PyTorch установлен правильно")
        logger.info("Запускаем обычное обучение...")
        return run_single_gpu_training()

def run_single_gpu_training():
    """Запуск обучения на одном GPU"""
    logger.info("Запуск обучения на одном GPU...")
    
    cmd = [
        sys.executable,
        "src/train.py",
        "--train_file=data/train_dataset.jsonl",
        "--test_file=data/train_dataset_test.jsonl",
        "--output_dir=models/legal_model",
        "--epochs=3",
        "--batch_size=1",
        "--gradient_accumulation_steps=8",
        "--learning_rate=1e-5",
        "--max_length=2048"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Обучение завершено успешно!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при обучении: {e}")

if __name__ == '__main__':
    run_distributed_training() 