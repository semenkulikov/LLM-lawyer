#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from openai import OpenAI
from openai import OpenAIError
from loguru import logger

# Загружаем переменные окружения из .env файла
def load_env():
    """Загрузка переменных окружения из .env файла"""
    env_file = Path('.env')
    if env_file.exists():
        logger.info("📄 Загрузка переменных окружения из .env файла...")
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Убираем кавычки из значения
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        logger.success("✅ Переменные окружения загружены")
    else:
        logger.warning("⚠️  Файл .env не найден")

# Загружаем переменные окружения при запуске
load_env()

def test_openai_key():
    """
    Тестирование OpenAI API ключа
    """
    # Получаем API ключ из переменной окружения
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("❌ Переменная окружения OPENAI_API_KEY не установлена")
        logger.info("Установите её командой:")
        logger.info("Windows: set OPENAI_API_KEY=sk-your-key-here")
        logger.info("Linux/Mac: export OPENAI_API_KEY=sk-your-key-here")
        return False
    
    # Проверяем формат ключа
    if not api_key.startswith("sk-"):
        logger.error("❌ Неверный формат API ключа. Ключ должен начинаться с 'sk-'")
        return False
    
    logger.info(f"🔑 API ключ найден: {api_key[:10]}...")
    
    # Создаем клиент
    try:
        client = OpenAI(api_key=api_key)
        logger.info("✅ Клиент OpenAI создан успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка создания клиента: {e}")
        return False
    
    # Тестируем API
    try:
        logger.info("🧪 Тестирование API...")
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Привет! Ответь одним словом: 'работает'"}],
            max_tokens=5,
        )
        
        response = completion.choices[0].message.content
        logger.info(f"✅ API работает! Ответ: {response}")
        
        # Проверяем баланс (если доступно)
        try:
            # Попытка получить информацию о балансе
            logger.info("💰 Проверка баланса...")
            # Примечание: OpenAI не предоставляет прямой API для проверки баланса
            # Можно использовать billing API если доступен
            logger.info("ℹ️  Для проверки баланса используйте: https://platform.openai.com/usage")
        except Exception as e:
            logger.warning(f"⚠️  Не удалось проверить баланс: {e}")
        
        return True
        
    except OpenAIError as e:
        if "authentication" in str(e).lower() or "invalid" in str(e).lower():
            logger.error("❌ Ошибка аутентификации: неверный API ключ")
        elif "rate_limit" in str(e).lower():
            logger.error("❌ Превышен лимит запросов (rate limit)")
        elif "quota" in str(e).lower() or "billing" in str(e).lower():
            logger.error("❌ Закончились средства на балансе")
        else:
            logger.error(f"❌ Ошибка OpenAI API: {e}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}")
        return False

def main():
    """
    Основная функция
    """
    logger.info("🚀 Тестирование OpenAI API ключа")
    logger.info("=" * 50)
    
    success = test_openai_key()
    
    logger.info("=" * 50)
    if success:
        logger.success("🎉 Тест пройден успешно! API ключ работает корректно.")
        logger.info("Теперь можно запускать основной пайплайн:")
        logger.info("python run_pipeline.py")
    else:
        logger.error("💥 Тест не пройден. Проверьте API ключ и повторите попытку.")
        sys.exit(1)

if __name__ == '__main__':
    main() 