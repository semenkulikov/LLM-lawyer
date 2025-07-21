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
        return f"""Ты - эксперт по анализу судебных документов. Проанализируй следующий текст судебного решения и извлеки структурированную информацию.

ТЕКСТ ДОКУМЕНТА:
{text}

ИНСТРУКЦИИ ПО АНАЛИЗУ:

1. ОПРЕДЕЛИ ТИП ДОКУМЕНТА:
   - Решение суда
   - Определение суда  
   - Постановление суда
   - Приговор суда
   - Другое (указать)

2. ИЗВЛЕКИ ОСНОВНУЮ ИНФОРМАЦИЮ:
   - Название суда
   - Номер дела
   - Дата вынесения
   - Судья (ФИО)
   - Стороны дела (истец/заявитель, ответчик/обвиняемый)

3. РАЗДЕЛИ НА СЕКЦИИ:
   - УСТАНОВОЧНАЯ ЧАСТЬ (фактические обстоятельства дела)
   - МОТИВИРОВОЧНАЯ ЧАСТЬ (правовая оценка, обоснование)
   - РЕЗОЛЮТИВНАЯ ЧАСТЬ (выводы и решения суда)

4. ОПРЕДЕЛИ ПРЕДМЕТ СПОРА:
   - Краткое описание сути спора
   - Основные требования сторон
   - Предметная область права (гражданское, уголовное, административное, налоговое, трудовое)

5. ВЫДЕЛИ КЛЮЧЕВЫЕ ФАКТЫ:
   - Основные обстоятельства дела
   - Важные даты и события
   - Представленные доказательства

6. ИЗВЛЕКИ ПРАВОВЫЕ НОРМЫ:
   - Примененные статьи законов
   - Ссылки на нормативные акты
   - Правовые принципы

7. ОПРЕДЕЛИ РЕЗУЛЬТАТ:
   - Решение суда
   - Обжалование (если упоминается)
   - Исполнение решения

ВЕРНИ ОТВЕТ В СЛЕДУЮЩЕМ JSON ФОРМАТЕ:

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
            logger.info(f"Анализирую документ: {filename}")
            
            # Создаем промпт
            prompt = self.create_analysis_prompt(text, filename)
            
            # Отправляем запрос к API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты - эксперт по анализу судебных документов. Отвечай строго в JSON формате."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Низкая температура для более точных ответов
                max_tokens=4000   # Ограничиваем токены для экономии
            )
            
            # Получаем ответ
            content = response.choices[0].message.content
            
            # Парсим JSON
            try:
                result = json.loads(content)
                logger.info(f"Успешно проанализирован документ: {filename}")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON для {filename}: {e}")
                logger.error(f"Полученный ответ: {content[:500]}...")
                return None
                
        except OpenAIError as e:
            logger.error(f"Ошибка OpenAI API для {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при анализе {filename}: {e}")
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
                
            logger.info(f"Результат сохранен: {output_path}")
            
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
                logger.info(f"Достигнут лимит документов ({self.max_documents}). Остановка обработки.")
                return False
            
            # Читаем файл
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"Файл {input_file} пустой, пропускаем")
                return False
            
            # Анализируем документ
            filename = Path(input_file).name
            result = self.analyze_document(text, filename)
            
            if result:
                # Сохраняем результат
                self.save_result(result, output_dir, filename)
                self.processed_count += 1
                
                # Небольшая пауза между запросами
                time.sleep(1)
                return True
            else:
                logger.error(f"Не удалось проанализировать документ: {input_file}")
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
            # Получаем список текстовых файлов
            text_files = list(Path(input_dir).glob("*.txt"))
            
            if not text_files:
                logger.warning(f"В директории {input_dir} не найдено текстовых файлов")
                return
            
            logger.info(f"Найдено {len(text_files)} текстовых файлов")
            logger.info(f"Будет обработано максимум {self.max_documents} файлов")
            
            # Обрабатываем файлы
            processed = 0
            for text_file in text_files:
                if self.process_text_file(str(text_file), output_dir):
                    processed += 1
                    logger.info(f"Обработано {processed}/{min(len(text_files), self.max_documents)} файлов")
                
                # Проверяем лимит
                if self.processed_count >= self.max_documents:
                    break
            
            logger.info(f"Обработка завершена. Успешно обработано {processed} файлов")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке директории {input_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Анализ судебных документов с помощью OpenAI API')
    parser.add_argument('--input-dir', type=str, required=True, help='Директория с текстовыми файлами')
    parser.add_argument('--output-dir', type=str, required=True, help='Директория для сохранения результатов')
    parser.add_argument('--api-key', type=str, help='OpenAI API ключ (или используй переменную окружения OPENAI_API_KEY)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Модель OpenAI для использования')
    parser.add_argument('--max-docs', type=int, default=3, help='Максимальное количество документов для обработки')
    
    args = parser.parse_args()
    
    # Получаем API ключ
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Не указан OpenAI API ключ. Используйте --api-key или переменную окружения OPENAI_API_KEY")
        return
    
    # Создаем процессор
    processor = LegalDocumentProcessor(api_key, args.model)
    processor.max_documents = args.max_docs
    
    # Обрабатываем документы
    processor.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 