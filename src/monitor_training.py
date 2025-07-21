#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path
from loguru import logger

def check_tensorboard_installed():
    """
    Проверка установки TensorBoard
    
    Returns:
        bool: True, если TensorBoard установлен, иначе False
    """
    try:
        import tensorboard
        return True
    except ImportError:
        return False

def start_tensorboard(log_dir, port=6006):
    """
    Запуск TensorBoard для мониторинга обучения
    
    Args:
        log_dir: Директория с логами TensorBoard
        port: Порт для запуска TensorBoard
        
    Returns:
        process: Процесс TensorBoard
    """
    cmd = ["tensorboard", "--logdir", log_dir, "--port", str(port)]
    logger.info(f"Запуск TensorBoard с командой: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )
    
    # Ожидание запуска TensorBoard
    time.sleep(3)
    
    # Проверка, что процесс запущен
    if process.poll() is not None:
        stderr = process.stderr.read()
        logger.error(f"Ошибка запуска TensorBoard: {stderr}")
        raise RuntimeError(f"Не удалось запустить TensorBoard: {stderr}")
    
    logger.info(f"TensorBoard запущен и доступен по адресу: http://localhost:{port}")
    return process

def open_tensorboard_in_browser(port=6006):
    """
    Открытие TensorBoard в браузере
    
    Args:
        port: Порт, на котором запущен TensorBoard
    """
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    logger.info(f"TensorBoard открыт в браузере по адресу: {url}")

def find_latest_logs(base_dir):
    """
    Поиск последней директории с логами
    
    Args:
        base_dir: Базовая директория для поиска
        
    Returns:
        str: Путь к последней директории с логами
    """
    try:
        # Поиск всех директорий с логами
        log_dirs = list(Path(base_dir).glob("**/logs"))
        
        if not log_dirs:
            # Если директорий с логами нет, используем базовую директорию
            return base_dir
        
        # Сортировка директорий по времени модификации
        latest_log_dir = max(log_dirs, key=lambda p: p.stat().st_mtime)
        return str(latest_log_dir)
    
    except Exception as e:
        logger.warning(f"Ошибка при поиске логов: {str(e)}")
        return base_dir

def main():
    parser = argparse.ArgumentParser(description='Запуск TensorBoard для мониторинга обучения модели')
    parser.add_argument('--log_dir', type=str, default='models', help='Директория с логами TensorBoard')
    parser.add_argument('--port', type=int, default=6006, help='Порт для запуска TensorBoard')
    parser.add_argument('--open_browser', action='store_true', help='Открыть TensorBoard в браузере')
    
    args = parser.parse_args()
    
    # Проверка, что TensorBoard установлен
    if not check_tensorboard_installed():
        logger.error("TensorBoard не установлен. Установите его с помощью команды 'pip install tensorboard'")
        return
    
    # Поиск последней директории с логами, если указана базовая директория
    log_dir = find_latest_logs(args.log_dir)
    logger.info(f"Использование директории с логами: {log_dir}")
    
    # Запуск TensorBoard
    process = start_tensorboard(log_dir, args.port)
    
    # Открытие TensorBoard в браузере, если указан флаг
    if args.open_browser:
        open_tensorboard_in_browser(args.port)
    
    try:
        # Ожидание прерывания пользователем
        logger.info("TensorBoard запущен. Нажмите Ctrl+C для завершения.")
        while process.poll() is None:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания. Завершение TensorBoard...")
    
    finally:
        # Завершение процесса TensorBoard
        if process.poll() is None:
            process.terminate()
            logger.info("TensorBoard завершен.")

if __name__ == '__main__':
    main() 