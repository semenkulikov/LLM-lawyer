import os
import json
import logging
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
from typing import Dict, List, Tuple, Any
import argparse
from tqdm import tqdm
import pathlib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Скачаем необходимые ресурсы nltk
def download_nltk_resources():
    """Скачивает необходимые ресурсы NLTK"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)  # Добавляем загрузку Open Multilingual WordNet
        logging.info("NLTK ресурсы успешно загружены")
    except Exception as e:
        logging.error(f"Ошибка при загрузке NLTK ресурсов: {e}")

# Загрузка стоп-слов для русского языка
def get_stopwords():
    """Возвращает расширенный список стоп-слов для русского языка"""
    try:
        russian_stopwords = set(stopwords.words('russian'))
        # Добавляем дополнительные стоп-слова
        additional_stopwords = {
            'который', 'также', 'это', 'этот', 'так', 'весь', 'свой', 'наш', 'самый', 
            'мой', 'являться', 'иметь', 'далее', 'один', 'два', 'три', 'четыре', 'пять',
            'согласно', 'т.д.', 'т.п.', 'т.к.', 'т.е.', 'указанный', 'составлять'
        }
        russian_stopwords.update(additional_stopwords)
        return russian_stopwords
    except Exception as e:
        logging.error(f"Ошибка при загрузке стоп-слов: {e}")
        return set()

def clean_text(text: str) -> str:
    """
    Очищает текст от лишних символов и приводит к единому формату
    
    Args:
        text: Исходный текст
    
    Returns:
        Очищенный текст
    """
    if not text:
        return ""
    
    # Заменяем повторяющиеся переносы строк на один
    text = re.sub(r'\n+', '\n', text)
    
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Заменяем множественные пробелы на один
    text = re.sub(r' +', ' ', text)
    
    # Удаляем символы номеров страниц (например, 51, 52, 53)
    text = re.sub(r'\n\d+\s*\n', '\n', text)
    
    # Удаляем ссылки на дела и решения
    text = re.sub(r'Определение № [А-Яа-я0-9\-]+', '', text)
    
    # Приводим к нижнему регистру
    text = text.lower()
    
    return text.strip()

def tokenize_text(text: str) -> List[str]:
    """
    Разбивает текст на токены
    
    Args:
        text: Очищенный текст
    
    Returns:
        Список токенов
    """
    if not text:
        return []
    
    # Простая токенизация по пробелам и знакам препинания
    # Это более надежная альтернатива word_tokenize для русского языка
    tokens = re.findall(r'\b\w+\b', text)
    
    # Фильтруем стоп-слова
    stopwords_list = get_stopwords()
    tokens = [token for token in tokens if token not in stopwords_list]
    
    # Фильтруем короткие токены (1-2 символа)
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens

def analyze_document(file_path: str) -> Dict[str, Any]:
    """
    Анализирует документ и извлекает ключевые параметры
    
    Args:
        file_path: Путь к JSON-файлу с документом
    
    Returns:
        Словарь с результатами анализа
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        filename = data.get('filename', os.path.basename(file_path))
        facts = clean_text(data.get('sections', {}).get('facts', ''))
        reasoning = clean_text(data.get('sections', {}).get('reasoning', ''))
        conclusion = clean_text(data.get('sections', {}).get('conclusion', ''))
        
        # Анализ каждой секции
        facts_tokens = tokenize_text(facts)
        reasoning_tokens = tokenize_text(reasoning)
        conclusion_tokens = tokenize_text(conclusion)
        
        # Выделение ключевых слов по частоте употребления (простейший подход)
        facts_freq = {}
        reasoning_freq = {}
        conclusion_freq = {}
        
        for token in facts_tokens:
            facts_freq[token] = facts_freq.get(token, 0) + 1
            
        for token in reasoning_tokens:
            reasoning_freq[token] = reasoning_freq.get(token, 0) + 1
            
        for token in conclusion_tokens:
            conclusion_freq[token] = conclusion_freq.get(token, 0) + 1
        
        # Выделяем топ-20 ключевых слов для каждой секции
        facts_keywords = sorted(facts_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        reasoning_keywords = sorted(reasoning_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        conclusion_keywords = sorted(conclusion_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Определение предметной области решения
        # Простая эвристика - проверка наличия ключевых слов для разных предметных областей
        subject_areas = {
            'гражданское право': ['договор', 'обязательство', 'собственность', 'сделка', 'имущество', 'наследство'],
            'уголовное право': ['приговор', 'обвиняемый', 'подсудимый', 'преступление', 'умысел', 'наказание'],
            'административное право': ['правонарушение', 'штраф', 'протокол', 'ответчик', 'административный'],
            'налоговое право': ['налог', 'сбор', 'налоговый', 'бюджет', 'декларация', 'доход'],
            'трудовое право': ['работник', 'работодатель', 'трудовой', 'увольнение', 'зарплата', 'отпуск']
        }
        
        # Объединяем все токены для определения предметной области
        all_tokens = facts_tokens + reasoning_tokens + conclusion_tokens
        
        # Подсчитываем частоту ключевых слов для каждой предметной области
        subject_scores = {}
        for subject, keywords in subject_areas.items():
            score = sum(1 for token in all_tokens if token in keywords)
            subject_scores[subject] = score
        
        # Определяем предметную область с наивысшим счетом
        subject_area = max(subject_scores.items(), key=lambda x: x[1])[0] if subject_scores else 'неопределено'
        
        # Создаем результат анализа
        analysis_result = {
            'filename': filename,
            'subject_area': subject_area,
            'facts_length': len(facts),
            'reasoning_length': len(reasoning),
            'conclusion_length': len(conclusion),
            'facts_keywords': facts_keywords,
            'reasoning_keywords': reasoning_keywords,
            'conclusion_keywords': conclusion_keywords,
            'total_tokens': len(all_tokens)
        }
        
        return analysis_result
    
    except Exception as e:
        logging.error(f"Ошибка при анализе документа {file_path}: {e}")
        return {
            'filename': os.path.basename(file_path),
            'error': str(e)
        }

def analyze_all_documents(input_dir: str, output_dir: str) -> None:
    """
    Анализирует все JSON-документы в директории и сохраняет результаты
    
    Args:
        input_dir: Путь к директории с JSON-файлами
        output_dir: Путь для сохранения результатов анализа
    """
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Получаем список всех JSON-файлов
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    logging.info(f"Найдено {len(json_files)} JSON-файлов для анализа")
    
    # Анализируем каждый файл
    all_results = []
    for file_path in tqdm(json_files, desc="Анализ документов"):
        result = analyze_document(file_path)
        all_results.append(result)
        
        # Сохраняем индивидуальный результат
        output_filename = os.path.basename(file_path).replace('.json', '_analysis.json')
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Сохраняем общий результат
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_documents': len(all_results),
            'documents': all_results
        }, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Анализ завершен. Результаты сохранены в {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Анализ текстов судебных решений')
    parser.add_argument('--input-dir', type=str, required=True, help='Директория с JSON-файлами')
    parser.add_argument('--output-dir', type=str, required=True, help='Директория для сохранения результатов анализа')
    args = parser.parse_args()
    
    # Скачиваем ресурсы NLTK, если они еще не загружены
    download_nltk_resources()
    
    # Анализируем все документы
    analyze_all_documents(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 