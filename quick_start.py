#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    print("=" * 80)
    print("🚀 ЮРИДИЧЕСКИЙ АССИСТЕНТ - АВТОМАТИЧЕСКАЯ УСТАНОВКА")
    print("=" * 80)
    print("Этот скрипт автоматически настроит проект для работы с 50,000+ документами")
    print("=" * 80)

def check_python_version():
    """Проверка версии Python"""
    print("🔍 Проверка версии Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Ошибка: Требуется Python 3.9 или выше")
        print(f"   Текущая версия: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_cuda():
    """Проверка CUDA"""
    print("🔍 Проверка CUDA...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA доступна")
            return True
        else:
            print("⚠️  CUDA не найдена, будет использован CPU (медленнее)")
            return False
    except FileNotFoundError:
        print("⚠️  CUDA не найдена, будет использован CPU (медленнее)")
        return False

def create_virtual_environment():
    """Создание виртуального окружения"""
    print("🔧 Создание виртуального окружения...")
    
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Виртуальное окружение уже существует")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("✅ Виртуальное окружение создано")
        return True
    except subprocess.CalledProcessError:
        print("❌ Ошибка создания виртуального окружения")
        return False

def install_requirements():
    """Установка зависимостей"""
    print("📦 Установка зависимостей...")
    
    # Определяем команду активации в зависимости от ОС
    if platform.system() == "Windows":
        pip_path = ".venv\\Scripts\\pip"
    else:
        pip_path = ".venv/bin/pip"
    
    try:
        # Обновляем pip
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print("✅ pip обновлен")
        
        # Устанавливаем PyTorch с CUDA (если доступна)
        if check_cuda():
            print("📦 Установка PyTorch с CUDA...")
            subprocess.run([
                pip_path, "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], check=True)
        else:
            print("📦 Установка PyTorch для CPU...")
            subprocess.run([pip_path, "install", "torch", "torchvision", "torchaudio"], check=True)
        
        # Устанавливаем остальные зависимости
        print("📦 Установка остальных зависимостей...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Все зависимости установлены")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        return False

def create_directories():
    """Создание необходимых директорий"""
    print("📁 Создание директорий...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/structured",
        "models",
        "logs",
        "results"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Директории созданы")

def test_installation():
    """Тестирование установки"""
    print("🧪 Тестирование установки...")
    
    if platform.system() == "Windows":
        python_path = ".venv\\Scripts\\python"
    else:
        python_path = ".venv/bin/python"
    
    try:
        # Тест CUDA
        result = subprocess.run([python_path, "check_cuda.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA тест пройден")
        else:
            print("⚠️  CUDA тест не пройден, но это не критично")
        
        # Тест основных модулей
        subprocess.run([python_path, "-c", "import torch; print('PyTorch OK')"], check=True)
        subprocess.run([python_path, "-c", "import transformers; print('Transformers OK')"], check=True)
        
        print("✅ Все тесты пройдены")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

def print_next_steps():
    """Инструкции по следующим шагам"""
    print("\n" + "=" * 80)
    print("🎉 УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 80)
    print("\n📋 Следующие шаги:")
    print("\n1. 📄 Подготовьте данные:")
    print("   - Поместите PDF документы в папку 'data/raw/'")
    print("   - Для 50,000+ документов рекомендуется использовать SSD")
    
    print("\n2. 🔑 Настройте OpenAI API (если нужно):")
    print("   - Создайте файл .env с OPENAI_API_KEY=your-key")
    
    print("\n3. 🚀 Запустите обучение:")
    print("   # Активируйте окружение:")
    if platform.system() == "Windows":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n   # Запустите полный пайплайн:")
    print("   python run_pipeline.py --epochs 10 --batch-size 1")
    
    print("\n4. 🖥️  Запустите GUI:")
    print("   python gui/app.py")
    
    print("\n📚 Дополнительная документация:")
    print("   - README.md - основная документация")
    print("   - GETTING_STARTED.md - подробные инструкции")
    print("   - INSTALLATION_GUIDE.md - руководство по установке")
    
    print("\n⚠️  Важные замечания:")
    print("   - Для 50,000+ документов требуется минимум 16GB RAM")
    print("   - Рекомендуется GPU с 8GB+ памяти")
    print("   - Обучение может занять несколько часов/дней")
    
    print("\n" + "=" * 80)

def main():
    print_banner()
    
    # Проверки
    if not check_python_version():
        return False
    
    if not create_virtual_environment():
        return False
    
    if not install_requirements():
        return False
    
    create_directories()
    
    if not test_installation():
        print("⚠️  Некоторые тесты не пройдены, но установка может работать")
    
    print_next_steps()
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Установка завершена успешно!")
        else:
            print("\n❌ Установка завершена с ошибками")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Установка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1) 