#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_api_connection():
    """Быстрый тест подключения к API"""
    print("🔍 БЫСТРАЯ ПРОВЕРКА ПОДКЛЮЧЕНИЯ К API")
    print("=" * 50)
    
    # Загружаем переменные окружения
    env_file = os.path.join(".env")
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    # Проверяем API ключи
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print(f"GEMINI_API_KEY: {'✅ Установлен' if gemini_key and gemini_key != 'your_gemini_api_key_here' else '❌ Не установлен'}")
    print(f"OPENAI_API_KEY: {'✅ Установлен' if openai_key and openai_key != 'your_openai_api_key_here' else '❌ Не установлен'}")
    
    # Тестируем Gemini
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        print("\n🔄 Тестирование Gemini...")
        try:
            from hybrid_processor import create_hybrid_processor
            processor = create_hybrid_processor(provider="gemini")
            response = processor.process_with_external_llm(
                local_response="Тест.",
                original_query="Тест",
                mode="polish"
            )
            print("✅ Gemini работает!")
        except Exception as e:
            print(f"❌ Gemini ошибка: {e}")
    else:
        print("\n⏭️ Gemini пропущен (нет API ключа)")
    
    # Тестируем OpenAI
    if openai_key and openai_key != 'your_openai_api_key_here':
        print("\n🔄 Тестирование OpenAI...")
        try:
            from hybrid_processor import create_hybrid_processor
            processor = create_hybrid_processor(provider="openai")
            response = processor.process_with_external_llm(
                local_response="Тест.",
                original_query="Тест",
                mode="polish"
            )
            print("✅ OpenAI работает!")
        except Exception as e:
            print(f"❌ OpenAI ошибка: {e}")
    else:
        print("\n⏭️ OpenAI пропущен (нет API ключа)")
    
    print("\n🏁 Проверка завершена!")

if __name__ == "__main__":
    test_api_connection()
