#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from typing import Optional, Dict, Any
from loguru import logger

def load_env():
    """Загрузка переменных окружения из .env файла"""
    env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

class HybridProcessor:
    """Универсальный гибридный процессор: локальная модель + внешний LLM (Gemini/OpenAI)"""
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, model: str = None):
        """
        Инициализация гибридного процессора
        
        Args:
            provider: Провайдер ("gemini" или "openai")
            api_key: API ключ (если не указан, загружается из .env)
            model: Модель для использования
        """
        load_env()
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.client = None
        
        # Установка модели по умолчанию
        if not self.model:
            if self.provider == "gemini":
                self.model = "gemini-2.0-flash-exp"
            elif self.provider == "openai":
                self.model = "gpt-4o-mini"
        
        # Загрузка API ключа из .env если не указан
        if not self.api_key:
            if self.provider == "gemini":
                self.api_key = os.getenv('GEMINI_API_KEY')
            elif self.provider == "openai":
                self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(f"{self.provider.upper()} API ключ не найден. Укажите в .env файле или передайте в конструктор.")
        
        self._init_client()
    
    def _init_client(self):
        """Инициализация клиента в зависимости от провайдера"""
        try:
            if self.provider == "gemini":
                self._init_gemini_client()
            elif self.provider == "openai":
                self._init_openai_client()
            else:
                raise ValueError(f"Неподдерживаемый провайдер: {self.provider}")
                
        except Exception as e:
            logger.error(f"Ошибка инициализации {self.provider} клиента: {e}")
            raise
    
    def _init_gemini_client(self):
        """Инициализация клиента Gemini"""
        try:
            from google import genai
            from google.genai import types
            
            # Создаем клиент с API ключом
            self.client = genai.Client(api_key=self.api_key)
            
            # Настраиваем для использования v1alpha API (для Ultra)
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(api_version='v1alpha')
            )
            
            logger.info(f"Gemini клиент инициализирован с моделью: {self.model}")
            
        except ImportError:
            raise ImportError("Установите google-genai: pip install google-genai")
    
    def _init_openai_client(self):
        """Инициализация клиента OpenAI"""
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=self.api_key)
            
            logger.info(f"OpenAI клиент инициализирован с моделью: {self.model}")
            
        except ImportError:
            raise ImportError("Установите openai: pip install openai")
    
    def process_with_external_llm(self, local_response: str, original_query: str, 
                                mode: str = "polish") -> str:
        """
        Обработка ответа локальной модели через внешний LLM
        
        Args:
            local_response: Ответ от локальной модели
            original_query: Исходный запрос пользователя
            mode: Режим обработки ("polish", "enhance", "verify")
            
        Returns:
            Обработанный ответ
        """
        try:
            if self.provider == "gemini":
                return self._process_with_gemini(local_response, original_query, mode)
            elif self.provider == "openai":
                return self._process_with_openai(local_response, original_query, mode)
            else:
                raise ValueError(f"Неподдерживаемый провайдер: {self.provider}")
                
        except Exception as e:
            logger.error(f"Ошибка при обработке через {self.provider}: {e}")
            # Возвращаем исходный ответ локальной модели в случае ошибки
            return local_response
    
    def _process_with_gemini(self, local_response: str, original_query: str, mode: str) -> str:
        """Обработка через Gemini"""
        try:
            if mode == "polish":
                prompt = self._create_polish_prompt(local_response, original_query)
            elif mode == "enhance":
                prompt = self._create_enhance_prompt(local_response, original_query)
            elif mode == "verify":
                prompt = self._create_verify_prompt(local_response, original_query)
            else:
                raise ValueError(f"Неизвестный режим: {mode}")
            
            response = self._call_gemini_api(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке через Gemini: {e}")
            return local_response
    
    def _process_with_openai(self, local_response: str, original_query: str, mode: str) -> str:
        """Обработка через OpenAI"""
        try:
            if mode == "polish":
                prompt = self._create_polish_prompt(local_response, original_query)
            elif mode == "enhance":
                prompt = self._create_enhance_prompt(local_response, original_query)
            elif mode == "verify":
                prompt = self._create_verify_prompt(local_response, original_query)
            else:
                raise ValueError(f"Неизвестный режим: {mode}")
            
            response = self._call_openai_api(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке через OpenAI: {e}")
            return local_response
    
    def _create_polish_prompt(self, local_response: str, original_query: str) -> str:
        """Создание промпта для полировки текста"""
        return f"""Ты - опытный юрист-редактор с глубокими знаниями российского законодательства. Твоя задача - улучшить и отполировать юридический текст, сохранив его суть и структуру.

ИСХОДНЫЙ ЗАПРОС КЛИЕНТА:
{original_query}

ЧЕРНОВИК ОТ ЛОКАЛЬНОЙ МОДЕЛИ:
{local_response}

ЗАДАЧА:
1. Исправь грамматические и стилистические ошибки
2. Улучши юридическую терминологию и ссылки на нормы права
3. Добавь недостающие ссылки на статьи законов (если уместно)
4. Сделай текст более структурированным и читаемым
5. Сохрани все важные факты и выводы
6. Используй актуальные нормы российского законодательства

ВАЖНО:
- Не меняй суть и выводы
- Сохрани структуру документа
- Используй официально-деловой стиль
- Если в черновике есть ошибки по существу - исправь их
- Убедись в корректности ссылок на ГК РФ, ГПК РФ, АПК РФ

ОТВЕТЬ ТОЛЬКО ОТПОЛИРОВАННЫМ ТЕКСТОМ, БЕЗ ДОПОЛНИТЕЛЬНЫХ КОММЕНТАРИЕВ."""

    def _create_enhance_prompt(self, local_response: str, original_query: str) -> str:
        """Создание промпта для расширения и улучшения текста"""
        return f"""Ты - опытный юрист-консультант с экспертизой в российском праве. Твоя задача - расширить и дополнить юридический документ, добавив недостающие элементы.

ИСХОДНЫЙ ЗАПРОС КЛИЕНТА:
{original_query}

БАЗОВЫЙ ТЕКСТ ОТ ЛОКАЛЬНОЙ МОДЕЛИ:
{local_response}

ЗАДАЧА:
1. Добавь вводную часть с указанием суда и сторон (если отсутствует)
2. Расширь мотивировочную часть дополнительными аргументами
3. Добавь ссылки на конкретные статьи ГК РФ, ГПК РФ, АПК РФ
4. Включи резолютивную часть с четкими выводами
5. Сделай документ более полным и профессиональным
6. Добавь ссылки на актуальную судебную практику (если уместно)

ВАЖНО:
- Сохрани все факты из исходного текста
- Добавь только релевантную информацию
- Используй актуальные нормы российского права
- Сделай документ готовым к использованию в суде
- Учти особенности процессуального законодательства

ОТВЕТЬ ТОЛЬКО РАСШИРЕННЫМ ДОКУМЕНТОМ, БЕЗ ДОПОЛНИТЕЛЬНЫХ КОММЕНТАРИЕВ."""

    def _create_verify_prompt(self, local_response: str, original_query: str) -> str:
        """Создание промпта для проверки и исправления ошибок"""
        return f"""Ты - опытный юрист-эксперт с глубокими знаниями российского законодательства. Твоя задача - проверить юридический документ на ошибки и исправить их.

ИСХОДНЫЙ ЗАПРОС КЛИЕНТА:
{original_query}

ТЕКСТ ДЛЯ ПРОВЕРКИ:
{local_response}

ЗАДАЧА:
1. Проверь правильность ссылок на нормы российского права
2. Исправь юридические ошибки и неточности
3. Проверь логику рассуждений и соответствие процессуальному законодательству
4. Убедись в корректности выводов
5. Исправь фактические ошибки
6. Проверь соответствие требованиям к форме документа

ВАЖНО:
- Если есть серьезные ошибки - исправь их
- Если текст в целом корректен - только отполируй
- Сохрани структуру и стиль
- Укажи на исправления в комментарии
- Учти особенности российского гражданского и арбитражного процесса

ОТВЕТЬ В ФОРМАТЕ:
ИСПРАВЛЕННЫЙ ТЕКСТ:
[текст]

КОММЕНТАРИЙ:
[краткое описание исправлений]"""

    def _call_gemini_api(self, prompt: str) -> str:
        """Вызов Gemini API"""
        try:
            from google.genai import types
            
            # Конфигурация для генерации
            config = types.GenerateContentConfig(
                max_output_tokens=4000,
                temperature=0.3,
                top_p=0.9,
                system_instruction="Ты - опытный юрист-эксперт с глубокими знаниями российского законодательства и судебной практики."
            )
            
            # Вызов API
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            return response.text.strip()
                
        except Exception as e:
            logger.error(f"Ошибка при вызове Gemini API: {e}")
            raise Exception(f"Ошибка Gemini API: {e}")

    def _call_openai_api(self, prompt: str) -> str:
        """Вызов OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты - опытный юрист-эксперт с глубокими знаниями российского законодательства и судебной практики."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.3,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"Ошибка при вызове OpenAI API: {e}")
            raise Exception(f"Ошибка OpenAI API: {e}")

def create_hybrid_processor(provider: str = "gemini", api_key: Optional[str] = None, model: str = None) -> HybridProcessor:
    """Фабричная функция для создания гибридного процессора"""
    return HybridProcessor(provider=provider, api_key=api_key, model=model)
