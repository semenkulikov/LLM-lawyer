#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import argparse
from loguru import logger
from tqdm import tqdm
import pickle

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
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        logger.success("✅ Переменные окружения загружены")
    else:
        logger.warning("⚠️  Файл .env не найден")

# Загружаем переменные окружения
load_env()

class AsyncLegalDocumentProcessor:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_concurrent: int = 5):
        """
        Инициализация асинхронного процессора документов
        
        Args:
            api_key: OpenAI API ключ
            model: Модель для использования
            max_concurrent: Максимальное количество одновременных запросов
        """
        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.processed_count = 0
        self.max_documents = 50000
        self.processed_files: Set[str] = set()
        self.progress_file = "processing_progress.pkl"
        
        # Инициализация без загрузки старого прогресса
        logger.info("🔄 Инициализация асинхронного процессора")
        
    def load_progress(self):
        """Загрузка прогресса обработки"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'rb') as f:
                    self.processed_files = pickle.load(f)
                logger.info(f"📊 Загружен прогресс: {len(self.processed_files)} уже обработанных файлов")
            except Exception as e:
                logger.warning(f"⚠️  Не удалось загрузить прогресс: {e}")
    
    def save_progress(self):
        """Сохранение прогресса обработки"""
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(self.processed_files, f)
            logger.info(f"💾 Прогресс сохранен: {len(self.processed_files)} файлов")
        except Exception as e:
            logger.error(f"❌ Не удалось сохранить прогресс: {e}")
    
    def create_analysis_prompt(self, text: str, filename: str) -> str:
        """
        Создает детальный промпт для анализа судебного документа
        
        Args:
            text: Текст документа
            filename: Имя файла
            
        Returns:
            Промпт для OpenAI
        """
        return f"""Ты - эксперт по анализу судебных документов с 20-летним опытом работы в Верховном Суде РФ. Твоя задача - максимально точно и детально проанализировать судебный документ и извлечь структурированную информацию для обучения нейросети.

ТЕКСТ ДОКУМЕНТА:
{text}

ВАЖНО: Анализируй документ максимально внимательно. Каждая деталь имеет значение для обучения модели.

ИНСТРУКЦИИ ПО АНАЛИЗУ:

1. ОПРЕДЕЛИ ТИП ДОКУМЕНТА:
   - Решение суда
   - Определение суда  
   - Постановление суда
   - Приговор суда
   - Другое (указать точно)

2. ИЗВЛЕКИ ОСНОВНУЮ ИНФОРМАЦИЮ:
   - Название суда (полное официальное название)
   - Номер дела (точно как указано)
   - Дата вынесения (в формате ДД.ММ.ГГГГ)
   - Судья (полное ФИО)
   - Стороны дела (истец/заявитель, ответчик/обвиняемый с полными ФИО)

3. РАЗДЕЛИ НА СЕКЦИИ (КРИТИЧЕСКИ ВАЖНО):
   - УСТАНОВОЧНАЯ ЧАСТЬ: фактические обстоятельства дела, доказательства, показания сторон
   - МОТИВИРОВОЧНАЯ ЧАСТЬ: правовая оценка, обоснование решения, ссылки на законы
   - РЕЗОЛЮТИВНАЯ ЧАСТЬ: выводы и решения суда

4. ОПРЕДЕЛИ ПРЕДМЕТ СПОРА:
   - Краткое описание сути спора (1-2 предложения)
   - Основные требования сторон (конкретно)
   - Предметная область права (гражданское, уголовное, административное, налоговое, трудовое, арбитражное)

5. ВЫДЕЛИ КЛЮЧЕВЫЕ ФАКТЫ:
   - Основные обстоятельства дела (хронологически)
   - Важные даты и события
   - Представленные доказательства
   - Ключевые аргументы сторон

6. ИЗВЛЕКИ ПРАВОВЫЕ НОРМЫ:
   - Примененные статьи законов (с номерами)
   - Ссылки на нормативные акты
   - Правовые принципы
   - Судебная практика (если упоминается)

7. ОПРЕДЕЛИ РЕЗУЛЬТАТ:
   - Решение суда (конкретно)
   - Обжалование (если упоминается)
   - Исполнение решения (если указано)

8. КАЧЕСТВО АНАЛИЗА:
   - Оценка полноты извлеченной информации
   - Уровень уверенности в анализе
   - Особые замечания

ВЕРНИ ОТВЕТ СТРОГО В СЛЕДУЮЩЕМ JSON ФОРМАТЕ (без лишних символов):

{{
    "document_info": {{
        "filename": "{filename}",
        "document_type": "тип документа",
        "court_name": "название суда",
        "case_number": "номер дела",
        "decision_date": "дата вынесения",
        "judge": "ФИО судьи",
        "parties": {{
            "plaintiff": "истец/заявитель",
            "defendant": "ответчик/обвиняемый"
        }}
    }},
    "sections": {{
        "factual_part": "установочная часть (фактические обстоятельства)",
        "reasoning_part": "мотивировочная часть (правовая оценка)",
        "operative_part": "резолютивная часть (решение суда)"
    }},
    "case_details": {{
        "dispute_subject": "предмет спора",
        "parties_claims": "требования сторон",
        "legal_area": "область права"
    }},
    "key_facts": [
        "факт 1",
        "факт 2",
        "факт 3"
    ],
    "legal_norms": [
        "статья закона 1",
        "статья закона 2"
    ],
    "court_decision": "решение суда",
    "analysis_quality": {{
        "completeness": "оценка полноты",
        "confidence": "уровень уверенности",
        "notes": "особые замечания"
    }}
}}"""

    async def analyze_document_async(self, session: aiohttp.ClientSession, text: str, filename: str, proxy_settings: str = None) -> Optional[Dict[str, Any]]:
        """
        Асинхронный анализ документа с помощью OpenAI API
        
        Args:
            session: aiohttp сессия
            text: Текст документа
            filename: Имя файла
            
        Returns:
            Результат анализа или None при ошибке
        """
        try:
            prompt = self.create_analysis_prompt(text, filename)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.1
            }
            
            # Добавляем прокси в запрос
            request_kwargs = {
                "headers": headers,
                "json": data,
                "timeout": aiohttp.ClientTimeout(total=120)
            }
            
            # Добавляем прокси если настроен
            if proxy_settings:
                request_kwargs["proxy"] = proxy_settings
            
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                **request_kwargs
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Парсим JSON ответ
                    try:
                        analysis_result = json.loads(content)
                        analysis_result["processing_info"] = {
                            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "model_used": self.model,
                            "filename": filename
                        }
                        return analysis_result
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Ошибка парсинга JSON для {filename}: {e}")
                        logger.error(f"📄 Полученный ответ: {content[:200]}...")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"❌ API ошибка для {filename}: {response.status} - {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"⏰ Таймаут при обработке {filename}")
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {filename}: {e}")
            return None

    def save_result(self, result: Dict[str, Any], output_dir: str, filename: str) -> None:
        """
        Сохранение результата анализа
        
        Args:
            result: Результат анализа
            output_dir: Директория для сохранения
            filename: Имя файла
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Создаем имя выходного файла
            base_name = Path(filename).stem
            output_file = os.path.join(output_dir, f"{base_name}_analyzed.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Результат сохранен: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результата для {filename}: {e}")

    async def process_text_file_async(self, session: aiohttp.ClientSession, input_file: str, output_dir: str) -> bool:
        """
        Асинхронная обработка одного текстового файла
        
        Args:
            session: aiohttp сессия
            input_file: Путь к входному файлу
            output_dir: Директория для сохранения
            
        Returns:
            True если успешно, False иначе
        """
        try:
            # Проверяем, не обработан ли уже файл
            if input_file in self.processed_files:
                logger.info(f"⏭️  Файл уже обработан, пропускаем: {input_file}")
                return True
            
            # Читаем файл
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            filename = os.path.basename(input_file)
            logger.info(f"📄 Обработка: {filename}")
            
            # Анализируем документ
            result = await self.analyze_document_async(session, text, filename)
            
            if result:
                # Сохраняем результат
                self.save_result(result, output_dir, filename)
                
                # Отмечаем как обработанный
                self.processed_files.add(input_file)
                # Прогресс обрабатывается автоматически
                
                self.processed_count += 1
                return True
            else:
                logger.warning(f"⚠️  Не удалось обработать: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке файла {input_file}: {e}")
            return False

    async def process_directory_async(self, input_dir: str, output_dir: str) -> None:
        """
        Асинхронная обработка всех текстовых файлов в директории с проверкой уже обработанных
        
        Args:
            input_dir: Директория с входными файлами
            output_dir: Директория для сохранения результатов
        """
        try:
            logger.info(f"🚀 Начинаем асинхронную обработку директории: {input_dir}")
            
            # Получаем список текстовых файлов
            logger.info(f"🔍 Поиск текстовых файлов...")
            text_files = list(Path(input_dir).glob("*.txt"))
            
            if not text_files:
                logger.warning(f"⚠️  В директории {input_dir} не найдено текстовых файлов")
                return
            
            # Проверяем уже обработанные файлы в выходной директории
            processed_files = set()
            if os.path.exists(output_dir):
                processed_files = {f for f in os.listdir(output_dir) if f.lower().endswith('_analyzed.json')}
                processed_files = {f.replace('_analyzed.json', '') for f in processed_files}
            
            logger.info(f"📁 Найдено {len(text_files)} текстовых файлов")
            logger.info(f"📊 Уже обработано: {len(processed_files)} файлов")
            
            # Фильтруем файлы для обработки
            files_to_process = []
            skipped_files = []
            
            for text_file in text_files:
                file_name = text_file.stem  # Имя файла без расширения
                output_file = os.path.join(output_dir, f"{file_name}_analyzed.json")
                
                if file_name in processed_files or os.path.exists(output_file):
                    skipped_files.append(str(text_file))
                else:
                    files_to_process.append(str(text_file))
            
            logger.info(f"📊 Пропущено (уже обработано): {len(skipped_files)} файлов")
            logger.info(f"📊 Будет обработано: {len(files_to_process)} файлов")
            logger.info(f"🎯 Будет обработано максимум {self.max_documents} файлов")
            
            # Ограничиваем количество файлов
            files_to_process = files_to_process[:self.max_documents]
            
            if not files_to_process:
                logger.info("✅ Все файлы уже обработаны!")
                return
            
            logger.info(f"🚀 Начинаем обработку {len(files_to_process)} файлов...")
            
            # Создаем семафор для ограничения одновременных запросов
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # Создаем одну общую сессию для всех задач с поддержкой прокси
            connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
            timeout = aiohttp.ClientTimeout(total=120)
            
            # Настройки прокси (если VPN использует локальный прокси)
            proxy_settings = None
            # Попробуем найти прокси в переменных окружения
            if os.getenv('HTTP_PROXY'):
                proxy_settings = os.getenv('HTTP_PROXY')
            elif os.getenv('HTTPS_PROXY'):
                proxy_settings = os.getenv('HTTPS_PROXY')
            # Обычные порты VPN прокси
            elif not proxy_settings:
                # Попробуем стандартные порты VPN
                for port in [12334, 1080, 8080, 3128, 8888]:
                    proxy_settings = f"http://127.0.0.1:{port}"
                    break  # Используем первый доступный
            
            if proxy_settings:
                logger.info(f"🔗 Используем прокси: {proxy_settings}")
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                connector_owner=False
            ) as session:
                async def process_with_semaphore(file_path):
                    async with semaphore:
                        return await self.process_text_file_async(session, file_path, output_dir, proxy_settings)
                
                # Обрабатываем файлы асинхронно
                tasks = [process_with_semaphore(file_path) for file_path in files_to_process]
                
                # Создаем прогресс-бар
                with tqdm(total=len(tasks), desc="📄 Асинхронная обработка", unit="док") as pbar:
                    completed = 0
                    for coro in asyncio.as_completed(tasks):
                        result = await coro
                        completed += 1
                        pbar.update(1)
                        
                        if result:
                            logger.info(f"✅ Успешно обработано: {completed}/{len(tasks)}")
                        else:
                            logger.warning(f"⚠️  Пропущен документ")
                
                logger.info(f"\n🎉 Асинхронная обработка завершена!")
                logger.info(f"📊 Итоговая статистика:")
                logger.info(f"   ✅ Успешно обработано: {self.processed_count} файлов")
                logger.info(f"   📁 Всего найдено: {len(text_files)} файлов")
                logger.info(f"   🎯 Лимит: {self.max_documents} файлов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при асинхронной обработке директории {input_dir}: {e}")

async def main():
    parser = argparse.ArgumentParser(description='Асинхронный анализ судебных документов с помощью OpenAI API')
    parser.add_argument('--input-dir', type=str, required=True, help='Директория с текстовыми файлами')
    parser.add_argument('--output-dir', type=str, required=True, help='Директория для сохранения результатов')
    parser.add_argument('--api-key', type=str, help='OpenAI API ключ (или используй переменную окружения OPENAI_API_KEY)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Модель OpenAI для использования')
    parser.add_argument('--max-docs', type=int, default=50000, help='Максимальное количество документов для обработки')
    parser.add_argument('--max-concurrent', type=int, default=5, help='Максимальное количество одновременных запросов')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🤖 ЗАПУСК АСИНХРОННОГО АНАЛИЗА СУДЕБНЫХ ДОКУМЕНТОВ С OPENAI")
    logger.info("="*80)
    logger.info(f"📁 Входная директория: {args.input_dir}")
    logger.info(f"📁 Выходная директория: {args.output_dir}")
    logger.info(f"🤖 Модель OpenAI: {args.model}")
    logger.info(f"🎯 Максимум документов: {args.max_docs}")
    logger.info(f"⚡ Максимум одновременных запросов: {args.max_concurrent}")
    logger.info("="*80)
    
    # Получаем API ключ
    logger.info("🔑 Проверка OpenAI API ключа...")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ Не указан OpenAI API ключ. Используйте --api-key или переменную окружения OPENAI_API_KEY")
        return
    
    logger.info("✅ OpenAI API ключ найден")
    
    # Создаем процессор
    logger.info("🔧 Инициализация асинхронного процессора документов...")
    processor = AsyncLegalDocumentProcessor(api_key, args.model, args.max_concurrent)
    processor.max_documents = args.max_docs
    logger.info("✅ Процессор инициализирован")
    
    # Обрабатываем документы
    logger.info("🚀 Начинаем асинхронную обработку документов...")
    start_time = time.time()
    
    # Обрабатываем документы
    await processor.process_directory_async(args.input_dir, args.output_dir)
    
    total_time = time.time() - start_time
    
    logger.info("="*80)
    logger.info("🎉 АСИНХРОННАЯ ОБРАБОТКА ЗАВЕРШЕНА!")
    logger.info("="*80)
    logger.info(f"⏱️  Общее время: {total_time:.2f} секунд ({total_time/60:.1f} минут)")
    logger.info(f"📊 Обработано документов: {processor.processed_count}")
    logger.info(f"🚀 Средняя скорость: {processor.processed_count/(total_time/60):.1f} документов/минуту")
    logger.info("="*80)

if __name__ == "__main__":
    asyncio.run(main())