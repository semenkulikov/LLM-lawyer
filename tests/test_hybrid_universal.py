#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference import load_model, generate
from hybrid_processor import create_hybrid_processor
from loguru import logger

def test_hybrid_universal():
    """Тестирование универсального гибридного подхода"""
    
    # Тестовый запрос
    test_facts = """
    Истец Иванов И.И. обратился в суд с иском к ответчику Петрову П.П. о взыскании задолженности по договору займа в размере 100 000 рублей. 
    Договор займа был заключен 15 января 2024 года, срок возврата - 15 марта 2024 года. 
    Ответчик задолженность не возвратил, несмотря на неоднократные требования истца.
    """
    
    print("🧪 ТЕСТИРОВАНИЕ УНИВЕРСАЛЬНОГО ГИБРИДНОГО ПОДХОДА")
    print("=" * 60)
    print(f"📝 Тестовый запрос:\n{test_facts}")
    print("=" * 60)
    
    try:
        # Шаг 1: Загрузка локальной модели
        print("🔄 Загрузка локальной модели QVikhr...")
        model_path = os.path.join("models", "legal_model")
        if not os.path.exists(model_path):
            print("❌ Модель не найдена! Сначала обучите модель.")
            return
        
        model, tokenizer = load_model(model_path)
        print("✅ Локальная модель QVikhr загружена")
        
        # Шаг 2: Генерация локальной моделью
        print("\n🔄 Генерация локальной моделью QVikhr...")
        local_response = generate(
            model=model,
            tokenizer=tokenizer,
            facts=test_facts,
            max_input_length=1024,
            max_output_length=1024
        )
        
        print("✅ Локальная модель QVikhr сгенерировала ответ")
        print(f"\n📄 ЛОКАЛЬНЫЙ РЕЗУЛЬТАТ (QVikhr):\n{local_response}")
        print("=" * 60)
        
        # Шаг 3: Тестирование разных провайдеров
        providers = ["gemini", "openai"]
        
        for provider in providers:
            print(f"\n🔄 Тестирование провайдера: {provider.upper()}")
            try:
                # Инициализация гибридного процессора
                hybrid_processor = create_hybrid_processor(provider=provider)
                print(f"✅ {provider.upper()} процессор инициализирован")
                
                # Тестирование разных режимов
                modes = ["polish", "enhance", "verify"]
                
                for mode in modes:
                    print(f"  🔄 Режим: {mode}")
                    try:
                        hybrid_response = hybrid_processor.process_with_external_llm(
                            local_response=local_response,
                            original_query=test_facts,
                            mode=mode
                        )
                        
                        print(f"  ✅ Режим '{mode}' обработан успешно через {provider.upper()}")
                        print(f"  📄 РЕЗУЛЬТАТ РЕЖИМА '{mode.upper()}' ({provider.upper()}):\n{hybrid_response[:200]}...")
                        print("  " + "-" * 50)
                        
                    except Exception as e:
                        print(f"  ❌ Ошибка в режиме '{mode}': {e}")
                
            except Exception as e:
                print(f"❌ Ошибка инициализации {provider.upper()} процессора: {e}")
                print(f"💡 Убедитесь, что в .env файле указан правильный {provider.upper()}_API_KEY")
        
        print("\n🎉 Тестирование завершено успешно!")
        print("💡 Универсальный гибридный подход готов к использованию!")
        
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
        logger.error(f"Ошибка тестирования: {e}")

def test_provider_connection(provider: str):
    """Тестирование подключения к конкретному провайдеру"""
    print(f"\n🔍 ТЕСТИРОВАНИЕ ПОДКЛЮЧЕНИЯ К {provider.upper()} API")
    print("=" * 50)
    
    try:
        hybrid_processor = create_hybrid_processor(provider=provider)
        
        # Простой тест
        test_response = hybrid_processor.process_with_external_llm(
            local_response="Тестовый ответ локальной модели QVikhr.",
            original_query="Тестовый запрос",
            mode="polish"
        )
        
        print(f"✅ Подключение к {provider.upper()} API работает")
        print(f"📄 Тестовый ответ от {provider.upper()}:\n{test_response}")
        
    except Exception as e:
        print(f"❌ Ошибка подключения к {provider.upper()} API: {e}")
        print("💡 Проверьте:")
        print(f"   • Правильность API ключа в .env файле")
        print("   • Доступность интернета")
        print("   • VPN (если требуется)")
        if provider == "gemini":
            print("   • Установку google-genai: pip install google-genai")
        elif provider == "openai":
            print("   • Установку openai: pip install openai")

def test_simple_connection():
    """Простой тест подключения без локальной модели"""
    print("\n🧪 ПРОСТОЙ ТЕСТ ПОДКЛЮЧЕНИЯ")
    print("=" * 40)
    
    providers = ["gemini", "openai"]
    
    for provider in providers:
        print(f"\n🔄 Тестирование {provider.upper()}...")
        try:
            # Создаем процессор
            processor = create_hybrid_processor(provider=provider)
            print(f"✅ {provider.upper()} процессор создан успешно")
            
            # Тестовые данные
            test_local_response = """
            Суд считает требования истца обоснованными. 
            Между сторонами был заключен договор займа. 
            Ответчик не возвратил заем в срок.
            """
            
            test_query = "Истец требует взыскать задолженность по договору займа"
            
            print(f"📝 Тестовый запрос: {test_query}")
            print(f"📄 Локальный ответ: {test_local_response.strip()}")
            
            # Тестируем режим polish
            print(f"\n🔄 Тестирование режима 'polish' через {provider.upper()}...")
            result = processor.process_with_external_llm(
                local_response=test_local_response,
                original_query=test_query,
                mode="polish"
            )
            
            print(f"✅ Режим 'polish' работает через {provider.upper()}!")
            print(f"📄 Результат:\n{result}")
            
        except ImportError as e:
            print(f"❌ Ошибка импорта: {e}")
            print("💡 Убедитесь, что все зависимости установлены")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("💡 Проверьте настройки API в .env файле")

def test_api_keys():
    """Проверка наличия API ключей"""
    print("\n🔑 ПРОВЕРКА API КЛЮЧЕЙ")
    print("=" * 30)
    
    # Загружаем переменные окружения
    env_file = os.path.join(".env")
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print(f"GEMINI_API_KEY: {'✅ Установлен' if gemini_key and gemini_key != 'your_gemini_api_key_here' else '❌ Не установлен'}")
    print(f"OPENAI_API_KEY: {'✅ Установлен' if openai_key and openai_key != 'your_openai_api_key_here' else '❌ Не установлен'}")
    
    if not gemini_key or gemini_key == 'your_gemini_api_key_here':
        print("💡 Установите GEMINI_API_KEY в .env файле")
    
    if not openai_key or openai_key == 'your_openai_api_key_here':
        print("💡 Установите OPENAI_API_KEY в .env файле")

if __name__ == "__main__":
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ УНИВЕРСАЛЬНОГО ГИБРИДНОГО ПОДХОДА")
    print("=" * 70)
    
    # Проверка API ключей
    test_api_keys()
    
    # Простой тест подключения
    test_simple_connection()
    
    # Тест подключения к провайдерам
    test_provider_connection("gemini")
    test_provider_connection("openai")
    
    # Полный тест гибридного подхода
    test_hybrid_universal()
    
    print("\n" + "=" * 70)
    print("🏁 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
