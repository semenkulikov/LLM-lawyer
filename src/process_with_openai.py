#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI
from openai import OpenAIError
import argparse
from loguru import logger
from tqdm import tqdm

class LegalDocumentProcessor:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Инициализация процессора документов
        
        Args:
            api_key: OpenAI API ключ
            model: Модель для использования
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.processed_count = 0
        self.max_documents = 3  # Ограничение для тестирования
        
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
        "date": "дата вынесения",
        "judge": "ФИО судьи",
        "parties": {{
            "plaintiff": "истец/заявитель",
            "defendant": "ответчик/обвиняемый"
        }}
    }},
    "subject_matter": {{
        "dispute_description": "краткое описание спора",
        "main_claims": "основные требования",
        "legal_area": "предметная область права"
    }},
    "sections": {{
        "facts": "установочная часть - фактические обстоятельства дела",
        "reasoning": "мотивировочная часть - правовая оценка и обоснование", 
        "conclusion": "резолютивная часть - выводы и решения суда"
    }},
    "key_facts": [
        "факт 1",
        "факт 2",
        "факт 3"
    ],
    "legal_norms": [
        "статья 1 закона",
        "статья 2 кодекса"
    ],
    "court_decision": {{
        "main_decision": "основное решение суда",
        "appeal_info": "информация об обжаловании",
        "enforcement": "информация об исполнении"
    }},
    "analysis_quality": {{
        "completeness": "полнота извлеченной информации (высокая/средняя/низкая)",
        "confidence": "уверенность в корректности анализа (высокая/средняя/низкая)",
        "notes": "дополнительные замечания"
    }}
}}

ВАЖНО:
- Если какая-то информация не найдена, укажи "не указано"
- Сохраняй оригинальные формулировки из документа
- Анализируй только предоставленный текст
- Отвечай строго в JSON формате без дополнительного текста
"""

    def analyze_document(self, text: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        Анализирует документ с помощью OpenAI API
        
        Args:
            text: Текст документа
            filename: Имя файла
            
        Returns:
            Результат анализа или None при ошибке
        """
        try:
            logger.info(f"🔍 Анализ документа: {filename}")
            logger.info(f"   📄 Размер текста: {len(text)} символов")
            
            # Создаем промпт
            logger.info(f"   📝 Создание промпта...")
            prompt = self.create_analysis_prompt(text, filename)
            logger.info(f"   📝 Промпт создан ({len(prompt)} символов)")
            
            # Отправляем запрос к OpenAI
            logger.info(f"   🤖 Отправка запроса к OpenAI (модель: {self.model})...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты - эксперт по анализу судебных документов. Отвечай строго в JSON формате."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Низкая температура для более точных ответов
                max_tokens=4000   # Ограничиваем токены для экономии
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"   ✅ Ответ получен за {elapsed_time:.2f} секунд")
            
            # Получаем ответ
            content = response.choices[0].message.content
            logger.info(f"   📊 Размер ответа: {len(content)} символов")
            
            # Парсим JSON
            logger.info(f"   🔧 Парсинг JSON...")
            try:
                result = json.loads(content)
                logger.info(f"   ✅ JSON успешно распарсен")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Ошибка парсинга JSON для {filename}: {e}")
                logger.error(f"   📄 Полученный ответ: {content[:500]}...")
                return None
                
        except OpenAIError as e:
            logger.error(f"❌ Ошибка OpenAI API для {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка при анализе {filename}: {e}")
            return None

    def save_result(self, result: Dict[str, Any], output_dir: str, filename: str) -> None:
        """
        Сохраняет результат анализа в файл
        
        Args:
            result: Результат анализа
            output_dir: Директория для сохранения
            filename: Имя файла
        """
        try:
            # Создаем директорию если не существует
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Формируем имя выходного файла
            output_filename = f"{Path(filename).stem}_analyzed.json"
            output_path = Path(output_dir) / output_filename
            
            # Сохраняем результат
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            logger.info(f"💾 Результат сохранен: {output_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении результата для {filename}: {e}")

    def process_text_file(self, input_file: str, output_dir: str) -> bool:
        """
        Обрабатывает один текстовый файл
        
        Args:
            input_file: Путь к входному файлу
            output_dir: Директория для сохранения результатов
            
        Returns:
            True если успешно, False иначе
        """
        try:
            # Проверяем лимит документов
            if self.processed_count >= self.max_documents:
                logger.info(f"🛑 Достигнут лимит документов ({self.max_documents}). Остановка обработки.")
                return False
            
            # Читаем файл
            logger.info(f"📖 Чтение файла: {Path(input_file).name}")
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"⚠️  Файл {input_file} пустой, пропускаем")
                return False
            
            logger.info(f"📄 Размер файла: {len(text)} символов")
            
            # Анализируем документ
            filename = Path(input_file).name
            result = self.analyze_document(text, filename)
            
            if result:
                # Сохраняем результат
                self.save_result(result, output_dir, filename)
                self.processed_count += 1
                
                # Небольшая пауза между запросами
                logger.info(f"⏳ Пауза 1 секунда перед следующим документом...")
                time.sleep(1)
                return True
            else:
                logger.error(f"❌ Не удалось проанализировать документ: {input_file}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {input_file}: {e}")
            return False

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Обрабатывает все текстовые файлы в директории
        
        Args:
            input_dir: Директория с входными файлами
            output_dir: Директория для сохранения результатов
        """
        try:
            logger.info(f"🚀 Начинаем обработку директории: {input_dir}")
            
            # Получаем список текстовых файлов
            logger.info(f"🔍 Поиск текстовых файлов...")
            text_files = list(Path(input_dir).glob("*.txt"))
            
            if not text_files:
                logger.warning(f"⚠️  В директории {input_dir} не найдено текстовых файлов")
                return
            
            logger.info(f"📁 Найдено {len(text_files)} текстовых файлов")
            logger.info(f"🎯 Будет обработано максимум {self.max_documents} файлов")
            logger.info(f"📊 Прогресс: 0/{min(len(text_files), self.max_documents)}")
            
            # Обрабатываем файлы с прогресс-баром
            processed = 0
            with tqdm(total=min(len(text_files), self.max_documents), 
                     desc="📄 Обработка документов", 
                     unit="док") as pbar:
                
                for i, text_file in enumerate(text_files):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"📄 Документ {i+1}/{min(len(text_files), self.max_documents)}: {text_file.name}")
                    logger.info(f"{'='*60}")
                    
                    if self.process_text_file(str(text_file), output_dir):
                        processed += 1
                        pbar.update(1)
                        logger.info(f"✅ Успешно обработано: {processed}/{min(len(text_files), self.max_documents)}")
                    else:
                        logger.warning(f"⚠️  Пропущен документ: {text_file.name}")
                    
                    # Проверяем лимит
                    if self.processed_count >= self.max_documents:
                        logger.info(f"🛑 Достигнут лимит документов, останавливаемся")
                        break
            
            logger.info(f"\n🎉 Обработка завершена!")
            logger.info(f"📊 Итоговая статистика:")
            logger.info(f"   ✅ Успешно обработано: {processed} файлов")
            logger.info(f"   📁 Всего найдено: {len(text_files)} файлов")
            logger.info(f"   🎯 Лимит: {self.max_documents} файлов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке директории {input_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Анализ судебных документов с помощью OpenAI API')
    parser.add_argument('--input-dir', type=str, required=True, help='Директория с текстовыми файлами')
    parser.add_argument('--output-dir', type=str, required=True, help='Директория для сохранения результатов')
    parser.add_argument('--api-key', type=str, help='OpenAI API ключ (или используй переменную окружения OPENAI_API_KEY)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Модель OpenAI для использования')
    parser.add_argument('--max-docs', type=int, default=3, help='Максимальное количество документов для обработки')
    parser.add_argument('--max-workers', type=int, default=1, help='Количество параллельных потоков (пока не используется)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🤖 ЗАПУСК АНАЛИЗА СУДЕБНЫХ ДОКУМЕНТОВ С OPENAI")
    logger.info("="*80)
    logger.info(f"📁 Входная директория: {args.input_dir}")
    logger.info(f"📁 Выходная директория: {args.output_dir}")
    logger.info(f"🤖 Модель OpenAI: {args.model}")
    logger.info(f"🎯 Максимум документов: {args.max_docs}")
    logger.info("="*80)
    
    # Получаем API ключ
    logger.info("🔑 Проверка OpenAI API ключа...")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ Не указан OpenAI API ключ. Используйте --api-key или переменную окружения OPENAI_API_KEY")
        return
    
    logger.info("✅ OpenAI API ключ найден")
    
    # Создаем процессор
    logger.info("🔧 Инициализация процессора документов...")
    processor = LegalDocumentProcessor(api_key, args.model)
    processor.max_documents = args.max_docs
    logger.info("✅ Процессор инициализирован")
    
    # Обрабатываем документы
    logger.info("🚀 Начинаем обработку документов...")
    start_time = time.time()
    
    processor.process_directory(args.input_dir, args.output_dir)
    
    total_time = time.time() - start_time
    logger.info("="*80)
    logger.info(f"🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    logger.info(f"⏱️  Общее время: {total_time:.2f} секунд")
    logger.info(f"📊 Среднее время на документ: {total_time/max(1, processor.processed_count):.2f} секунд")
    logger.info("="*80)

if __name__ == "__main__":
    main() 