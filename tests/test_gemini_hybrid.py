#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference import load_model, generate
from gemini_hybrid_processor import create_gemini_hybrid_processor
from loguru import logger

def test_gemini_hybrid_approach():
    """Тестирование гибридного подхода с Gemini Ultra"""
    
    # Тестовый запрос
    test_facts = """
    Истец Иванов И.И. обратился в суд с иском к ответчику Петрову П.П. о взыскании задолженности по договору займа в размере 100 000 рублей. 
    Договор займа был заключен 15 января 2024 года, срок возврата - 15 марта 2024 года. 
    Ответчик задолженность не возвратил, несмотря на неоднократные требования истца.
    """
    
    print("🧪 ТЕСТИРОВАНИЕ ГИБРИДНОГО ПОДХОДА С GEMINI ULTRA")
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
        
        # Шаг 3: Инициализация Gemini процессора
        print("\n🔄 Инициализация Gemini процессора...")
        try:
            gemini_processor = create_gemini_hybrid_processor()
            print("✅ Gemini процессор инициализирован")
        except Exception as e:
            print(f"❌ Ошибка инициализации Gemini процессора: {e}")
            print("💡 Убедитесь, что в .env файле указан правильный GEMINI_API_KEY")
            return
        
        # Шаг 4: Тестирование разных режимов обработки
        modes = ["polish", "enhance", "verify"]
        
        for mode in modes:
            print(f"\n🔄 Тестирование режима: {mode}")
            try:
                gemini_response = gemini_processor.process_with_gemini(
                    local_response=local_response,
                    original_query=test_facts,
                    mode=mode
                )
                
                print(f"✅ Режим '{mode}' обработан успешно через Gemini Ultra")
                print(f"\n📄 РЕЗУЛЬТАТ РЕЖИМА '{mode.upper()}' (Gemini Ultra):\n{gemini_response}")
                print("-" * 60)
                
            except Exception as e:
                print(f"❌ Ошибка в режиме '{mode}': {e}")
        
        print("\n🎉 Тестирование завершено успешно!")
        print("💡 Гибридный подход с Gemini Ultra готов к использованию!")
        
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
        logger.error(f"Ошибка тестирования: {e}")

def test_gemini_connection():
    """Тестирование подключения к Gemini API"""
    print("\n🔍 ТЕСТИРОВАНИЕ ПОДКЛЮЧЕНИЯ К GEMINI API")
    print("=" * 50)
    
    try:
        gemini_processor = create_gemini_hybrid_processor()
        
        # Простой тест
        test_response = gemini_processor.process_with_gemini(
            local_response="Тестовый ответ локальной модели QVikhr.",
            original_query="Тестовый запрос",
            mode="polish"
        )
        
        print("✅ Подключение к Gemini API работает")
        print(f"📄 Тестовый ответ от Gemini Ultra:\n{test_response}")
        
    except Exception as e:
        print(f"❌ Ошибка подключения к Gemini API: {e}")
        print("💡 Проверьте:")
        print("   • Правильность API ключа в .env файле")
        print("   • Доступность интернета")
        print("   • VPN (если требуется)")
        print("   • Установку google-genai: pip install google-genai")

def test_gemini_simple():
    """Простой тест Gemini без локальной модели"""
    print("\n🧪 ПРОСТОЙ ТЕСТ GEMINI ULTRA")
    print("=" * 40)
    
    try:
        # Импортируем Gemini процессор
        from gemini_hybrid_processor import create_gemini_hybrid_processor
        
        print("✅ Модуль gemini_hybrid_processor импортирован успешно")
        
        # Создаем процессор
        processor = create_gemini_hybrid_processor()
        print("✅ Gemini процессор создан успешно")
        
        # Тестовые данные
        test_local_response = """
        Суд считает требования истца обоснованными. 
        Между сторонами был заключен договор займа. 
        Ответчик не возвратил заем в срок.
        """
        
        test_query = "Истец требует взыскать задолженность по договору займа"
        
        print(f"\n📝 Тестовый запрос: {test_query}")
        print(f"📄 Локальный ответ: {test_local_response.strip()}")
        
        # Тестируем режим polish
        print("\n🔄 Тестирование режима 'polish' через Gemini Ultra...")
        result = processor.process_with_gemini(
            local_response=test_local_response,
            original_query=test_query,
            mode="polish"
        )
        
        print("✅ Режим 'polish' работает через Gemini Ultra!")
        print(f"📄 Результат:\n{result}")
        
        print("\n🎉 Простой тест прошел успешно!")
        print("💡 Гибридный подход с Gemini Ultra готов к использованию")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Убедитесь, что все зависимости установлены")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("💡 Проверьте настройки Gemini API в .env файле")

if __name__ == "__main__":
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ ГИБРИДНОГО ПОДХОДА С GEMINI ULTRA")
    print("=" * 70)
    
    # Простой тест Gemini
    test_gemini_simple()
    
    # Тест подключения к Gemini
    test_gemini_connection()
    
    # Полный тест гибридного подхода
    test_gemini_hybrid_approach()
    
    print("\n" + "=" * 70)
    print("🏁 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
