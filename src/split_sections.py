import os
import json
import logging
import re
import pathlib
import traceback
import nltk
from typing import Dict, List, Tuple, Optional, Any
import argparse
from tqdm import tqdm
from collections import Counter

# Настройка логирования с более подробным уровнем
logging.basicConfig(
    level=logging.INFO,  # Изменяем на INFO для стандартного вывода
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Проверка и загрузка необходимых ресурсов NLTK
def download_nltk_resources():
    """
    Проверяет наличие и при необходимости загружает требуемые ресурсы NLTK
    """
    required_resources = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger'
    ]
    
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logging.info(f"Ресурс NLTK {resource} уже установлен")
        except LookupError:
            try:
                logging.info(f"Загрузка ресурса NLTK {resource}...")
                nltk.download(resource, quiet=True)
                logging.info(f"Ресурс NLTK {resource} успешно загружен")
            except Exception as e:
                logging.error(f"Ошибка при загрузке ресурса NLTK {resource}: {str(e)}")
                # Продолжаем работу даже при ошибке загрузки

# Загружаем необходимые ресурсы при импорте модуля
try:
    download_nltk_resources()
except Exception as e:
    logging.warning(f"Ошибка при загрузке ресурсов NLTK: {str(e)}")

# Улучшаем словарь маркеров, добавляя больше правовых терминов и формулировок
SECTION_MARKERS = {
    'facts': [
        # Стандартные маркеры установочной части
        r'(?i)(установил|УСТАНОВИЛ):?\s+',
        r'(?i)(суд\s+установил):?\s+',
        r'(?i)(фактические\s+обстоятельства):?\s+',
        # Дополнительные маркеры
        r'(?i)(обстоятельства\s+дела):?\s+',
        r'(?i)(описательная\s+часть):?\s+',
        r'(?i)(материалами\s+дела\s+установлено):?\s+',
        r'(?i)(из\s+материалов\s+дела\s+следует):?\s+',
        r'(?i)(в\s+судебном\s+заседании\s+установлено):?\s+',
        r'(?i)(в\s+ходе\s+рассмотрения\s+дела\s+установлено):?\s+',
        r'(?i)(установлено\s+следующее):?\s+',
        # Новые маркеры
        r'(?i)(в\s+обоснование\s+заявленных\s+требований):?\s+',
        r'(?i)(заявитель\s+обратился):?\s+',
        r'(?i)(истец\s+обратился):?\s+',
        # Дополнительные маркеры для фактов
        r'(?i)(заявитель\s+[а-я]+ет,\s+что):?\s+',
        r'(?i)(материалами\s+дела\s+подтверждается):?\s+',
        r'(?i)(из\s+материалов\s+дела\s+усматривается):?\s+',
        r'(?i)(суд\s+установил\s+следующее):?\s+',
        r'(?i)(в\s+судебное\s+заседание\s+[а-я]+\s+представил):?\s+',
        r'(?i)(как\s+следует\s+из\s+материалов\s+дела):?\s+',
        r'(?i)(в\s+материалах\s+дела\s+имеются):?\s+',
        r'(?i)(суд\s+принимает\s+во\s+внимание\s+следующее):?\s+',
        r'(?i)(в\s+заявлении\s+[а-я]+\s+указывает):?\s+',
        r'(?i)(в\s+судебном\s+заседании\s+установлено\s+следующее):?\s+'
    ],
    'reasoning': [
        # Стандартные маркеры мотивировочной части
        r'(?i)(мотивировочная\s+часть):?\s+',
        r'(?i)(руководствуясь):?\s+',
        r'(?i)(суд\s+считает):?\s+',
        r'(?i)(оценивая\s+представленные\s+доказательства):?\s+',
        # Дополнительные маркеры
        r'(?i)(исследовав\s+материалы\s+дела):?\s+',
        r'(?i)(проанализировав\s+доводы\s+сторон):?\s+',
        r'(?i)(суд\s+приходит\s+к\s+выводу):?\s+',
        r'(?i)(оценив\s+доказательства):?\s+',
        r'(?i)(обсудив\s+доводы\s+жалобы):?\s+',
        r'(?i)(в\s+соответствии\s+со\s+статьей):?\s+',
        r'(?i)(в\s+силу\s+статьи):?\s+',
        r'(?i)(согласно\s+статье):?\s+',
        r'(?i)(рассмотрев\s+материалы\s+дела):?\s+',
        # Новые маркеры
        r'(?i)(суд\s+полагает):?\s+',
        r'(?i)(правовая\s+оценка):?\s+',
        r'(?i)(оценка\s+суда):?\s+',
        r'(?i)(дав\s+оценку):?\s+',
        r'(?i)(учитывая\s+изложенное):?\s+',
        r'(?i)(принимая\s+во\s+внимание):?\s+',
        # Дополнительные маркеры для мотивировочной части
        r'(?i)(суд\s+руководствуется\s+следующим):?\s+',
        r'(?i)(изучив\s+материалы\s+дела):?\s+',
        r'(?i)(проанализировав\s+имеющиеся\s+доказательства):?\s+',
        r'(?i)(суд\s+считает\s+необходимым):?\s+',
        r'(?i)(суд\s+приходит\s+к\s+следующим\s+выводам):?\s+',
        r'(?i)(доводы\s+[а-я]+\s+о\s+том,\s+что):?\s+',
        r'(?i)(в\s+соответствии\s+с\s+[а-я]+):?\s+',
        r'(?i)(по\s+смыслу\s+[а-я]+):?\s+',
        r'(?i)(таким\s+образом,\s+суд\s+полагает):?\s+',
        r'(?i)(совокупность\s+представленных\s+доказательств):?\s+',
        r'(?i)(оценив\s+в\s+совокупности\s+все\s+доказательства):?\s+',
        r'(?i)(судом\s+установлено,\s+что):?\s+'
    ],
    'conclusion': [
        # Стандартные маркеры резолютивной части
        r'(?i)(резолютивная\s+часть):?\s+',
        r'(?i)(постановил|ПОСТАНОВИЛ):?\s+',
        r'(?i)(решил|РЕШИЛ):?\s+',
        r'(?i)(приговорил|ПРИГОВОРИЛ):?\s+',
        # Дополнительные маркеры
        r'(?i)(определил|ОПРЕДЕЛИЛ):?\s+',
        r'(?i)(на\s+основании\s+изложенного):?\s+',
        r'(?i)(руководствуясь\s+статьями):?\s+',
        r'(?i)(руководствуясь\s+ст\.):?\s+',
        r'(?i)(суд\s+постановляет):?\s+',
        r'(?i)(суд\s+определяет):?\s+',
        r'(?i)(суд\s+решает):?\s+',
        r'(?i)(руководствуясь\s+ст\.\s+\d+):?\s+',
        # Новые маркеры
        r'(?i)(руководствуясь\s+статьями[\s\d,]+):?\s+',
        r'(?i)(на\s+основании\s+ст\.[\s\d,]+):?\s+',
        r'(?i)(суд\s+постановил):?\s+',
        r'(?i)(по\s+результатам\s+рассмотрения):?\s+',
        r'(?i)(с\s+учетом\s+изложенного):?\s+',
        # Дополнительные маркеры для резолютивной части
        r'(?i)(руководствуясь\s+[а-я ]+,\s+суд):?\s+',
        r'(?i)(руководствуясь\s+статьями[\s\d,.]+[а-я]+):?\s+',
        r'(?i)(с\s+учетом\s+изложенного\s+и\s+руководствуясь):?\s+',
        r'(?i)(в\s+соответствии\s+со\s+статьями[\s\d,.]+[а-я]+):?\s+',
        r'(?i)(на\s+основании\s+изложенного\s+и\s+руководствуясь):?\s+',
        r'(?i)(исходя\s+из\s+изложенного\s+и\s+руководствуясь):?\s+',
        r'(?i)(руководствуясь\s+статьями[\s\d,.]+(,)?\s+суд\s+[а-я]+):?\s+',
        r'(?i)(учитывая\s+изложенное\s+и\s+руководствуясь):?\s+',
        r'(?i)(в\s+связи\s+с\s+изложенным,\s+суд):?\s+'
    ]
}

# Типичные структуры различных судебных документов
DOCUMENT_TYPES = {
    'civil': {
        'markers': [r'гражданское\s+дело', r'гражданский\s+иск', r'истец', r'ответчик'],
        'sections_order': ['facts', 'reasoning', 'conclusion']
    },
    'criminal': {
        'markers': [r'уголовное\s+дело', r'обвиняемый', r'подсудимый', r'преступление'],
        'sections_order': ['facts', 'reasoning', 'conclusion']
    },
    'administrative': {
        'markers': [r'административное\s+дело', r'административное\s+правонарушение', r'протокол'],
        'sections_order': ['facts', 'reasoning', 'conclusion']
    },
    'arbitration': {
        'markers': [r'арбитражный\s+суд', r'экономический\s+спор', r'заявитель'],
        'sections_order': ['facts', 'reasoning', 'conclusion']
    },
    'bulletin': {
        'markers': [r'бюллетень', r'бюллетень\s+верховного\s+суда', r'обзор\s+судебной\s+практики'],
        'sections_order': ['facts', 'reasoning', 'conclusion']
    },
    'review': {
        'markers': [r'обзор', r'обзор\s+практики', r'обзор\s+по\s+вопросам', r'разъяснения\s+по\s+вопросам'],
        'sections_order': ['facts', 'reasoning', 'conclusion']
    }
}

def detect_document_type(text: str) -> str:
    """
    Определяет тип судебного документа на основе характерных маркеров.
    
    Args:
        text: Текст документа
        
    Returns:
        Тип документа или 'unknown'
    """
    # Проверяем имя файла в тексте (иногда оно присутствует в первых строках)
    first_500_chars = text[:500].lower()
    
    # Проверяем специальные случаи по первым символам
    if 'бюллетень' in first_500_chars:
        return 'bulletin'
    elif 'обзор судебной практики' in first_500_chars:
        return 'review'
    
    # Подсчитываем маркеры для каждого типа документа
    type_scores = {}
    
    for doc_type, type_info in DOCUMENT_TYPES.items():
        score = 0
        for marker in type_info['markers']:
            matches = len(re.findall(marker, text, re.IGNORECASE))
            score += matches
        type_scores[doc_type] = score
    
    # Определяем тип с наибольшим количеством совпадений
    if all(score == 0 for score in type_scores.values()):
        return 'unknown'
    
    return max(type_scores.items(), key=lambda x: x[1])[0]

def split_into_sections(text: str) -> Dict[str, str]:
    """
    Разделяет текст на смысловые секции (факты, мотивировка, заключение).
    
    Args:
        text: Текст для разделения
        
    Returns:
        Словарь с текстами для каждой секции
    """
    # Определяем тип документа на основе его содержания
    doc_type = detect_document_type(text)
    
    # Словарь для хранения секций
    sections = {
        'facts': '',
        'reasoning': '',
        'conclusion': ''
    }
    
    # Специальная обработка для бюллетеней и обзоров
    if doc_type in ['bulletin', 'review']:
        try:
            # Для бюллетеней и обзоров используем семантический анализ
            semantic_sections = identify_semantic_parts(text)
            
            for section_name, parts in semantic_sections.items():
                if parts:
                    # Объединяем все найденные части секции
                    section_text = ""
                    for start, end in parts:
                        section_text += text[start:end] + "\n\n"
                    sections[section_name] = section_text.strip()
            
            # Дополнительная проверка и балансировка для бюллетеней
            sections = balance_section_lengths(sections, text)
            
            # Если все секции заполнены и их качество хорошее, возвращаем результат
            quality = assess_sections_quality(sections)
            if quality != 'низкое' and all(sections.values()):
                return sections
        except Exception as e:
            logging.warning(f"Ошибка при обработке документа типа {doc_type}: {str(e)}")
    
    # Стандартный алгоритм для других типов документов
    # Пытаемся найти границы секций на основе маркеров
    bounds = find_section_bounds(text, doc_type)
    
    # Если нашли хотя бы одну границу, извлекаем секции
    if any(idx is not None for idx in bounds.values()):
        sections = extract_sections_by_bounds(text, bounds)
    
    # Если не удалось выделить все секции с помощью маркеров,
    # применяем дополнительные эвристики
    if not all(sections.values()) or any(len(section) < 200 for section in sections.values() if section):
        sections = apply_extended_heuristics(text, sections, doc_type)
    
    # Анализируем соотношение длин секций и при необходимости 
    # корректируем их дополнительно
    sections = balance_section_lengths(sections, text)
    
    # Проверяем качество разделения
    quality = assess_sections_quality(sections)
    if quality == 'низкое':
        # Если качество низкое, пробуем применить дополнительные методы
        try:
            nlp_sections = apply_nlp_heuristics(text, sections)
            # Проверяем, улучшилось ли качество
            nlp_quality = assess_sections_quality(nlp_sections)
            if nlp_quality != 'низкое' or not all(sections.values()):
                sections = nlp_sections
        except Exception as e:
            logging.warning(f"Ошибка при применении NLP эвристик: {str(e)}")
    
    # Убедимся, что все секции заполнены
    if not sections['facts']:
        # Если факты не найдены, используем первую треть текста
        sections['facts'] = text[:len(text)//3].strip()
    
    if not sections['reasoning']:
        # Если мотивировка не найдена, используем середину текста
        start = len(text)//3
        end = 2*len(text)//3
        sections['reasoning'] = text[start:end].strip()
    
    if not sections['conclusion']:
        # Если заключение не найдено, используем последнюю треть текста
        sections['conclusion'] = text[2*len(text)//3:].strip()
    
    return sections

def find_section_bounds(text: str, doc_type: str) -> Dict[str, int]:
    """
    Находит индексы начала каждой секции в тексте.
    
    Args:
        text: Текст документа
        doc_type: Тип документа
        
    Returns:
        Словарь с индексами начала каждой секции
    """
    bounds = {
        'facts': None,
        'reasoning': None,
        'conclusion': None
    }
    
    # Используем более надежный поиск с учетом порядка секций
    sections_order = DOCUMENT_TYPES[doc_type]['sections_order'] if doc_type in DOCUMENT_TYPES else ['facts', 'reasoning', 'conclusion']
    
    # Для бюллетеней и обзоров используем другой подход
    if doc_type in ['bulletin', 'review']:
        # Для бюллетеней пытаемся найти разделы по номерам или специальным заголовкам
        numbers_pattern = r'\b(?:(\d+)\.|\((\d+)\))\s+([А-Я])'
        
        matches = list(re.finditer(numbers_pattern, text))
        if matches and len(matches) >= 3:
            # Используем первое, среднее и последнее совпадение
            first_match = matches[0]
            middle_match = matches[len(matches)//2]
            last_match = matches[-3] if len(matches) > 3 else matches[-1]
            
            bounds['facts'] = first_match.start()
            bounds['reasoning'] = middle_match.start()
            bounds['conclusion'] = last_match.start()
            
            return bounds
    
    # Ищем каждую секцию, начиная с начала документа
    current_pos = 0
    
    # Сначала ищем явные маркеры начала каждой секции
    for section in sections_order:
        earliest_pos = float('inf')
        earliest_marker = None
        
        for marker in SECTION_MARKERS[section]:
            match = re.search(marker, text[current_pos:], re.DOTALL)
            if match:
                pos = current_pos + match.start()
                if pos < earliest_pos:
                    earliest_pos = pos
                    earliest_marker = marker
        
        if earliest_pos < float('inf'):
            bounds[section] = earliest_pos
            match = re.search(earliest_marker, text[earliest_pos:], re.DOTALL)
            if match:
                current_pos = earliest_pos + match.end()
    
    # Если не нашли все секции, пытаемся использовать более гибкий подход
    # Часто в тексте может отсутствовать явный маркер начала фактов
    if bounds['facts'] is None and (bounds['reasoning'] is not None or bounds['conclusion'] is not None):
        # Для документов типа 'civil' и 'arbitration' обычно факты в начале документа
        if doc_type in ['civil', 'arbitration']:
            bounds['facts'] = 0  # Предполагаем, что факты начинаются с начала документа
        else:
            # Для других типов пытаемся найти начало фактов по косвенным признакам
            facts_start = find_facts_section_start(text)
            if facts_start is not None:
                bounds['facts'] = facts_start
            else:
                bounds['facts'] = 0
    
    # Проверяем правильный порядок секций
    valid_indices = [idx for idx in [bounds[section] for section in sections_order] if idx is not None]
    if valid_indices != sorted(valid_indices):
        # Если порядок секций не соответствует ожидаемому, пытаемся исправить
        try:
            corrected_bounds = correct_section_order(bounds, sections_order, text)
            # Проверяем, удалось ли исправить порядок
            corrected_indices = [idx for idx in [corrected_bounds[section] for section in sections_order] if idx is not None]
            if corrected_indices == sorted(corrected_indices):
                return corrected_bounds
        except Exception as e:
            logging.warning(f"Ошибка при исправлении порядка секций: {str(e)}")
        
        # Если не удалось исправить порядок, сбрасываем индексы
        logging.warning("Найденный порядок секций не соответствует ожидаемому")
        return {section: None for section in bounds}
    
    return bounds

def find_facts_section_start(text: str) -> Optional[int]:
    """
    Пытается найти начало секции фактов по косвенным признакам.
    
    Args:
        text: Текст документа
        
    Returns:
        Индекс начала секции фактов или None
    """
    # Ищем типичные начала для разных типов документов
    patterns = [
        r'(?i)([А-Я][а-я]+\s+суд\s+[а-я]+):?\s+',
        r'(?i)(Дело\s+№\s+[\d-А-Я]+)',
        r'(?i)(Р\s*Е\s*Ш\s*Е\s*Н\s*И\s*Е)',
        r'(?i)(О\s*П\s*Р\s*Е\s*Д\s*Е\s*Л\s*Е\s*Н\s*И\s*Е)',
        r'(?i)(П\s*О\s*С\s*Т\s*А\s*Н\s*О\s*В\s*Л\s*Е\s*Н\s*И\s*Е)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Находим конец этого заголовка и следующий абзац
            line_end = text.find('\n', match.end())
            if line_end > 0:
                next_non_empty = re.search(r'[^\s]', text[line_end:])
                if next_non_empty:
                    return line_end + next_non_empty.start()
    
    return None

def correct_section_order(bounds: Dict[str, Optional[int]], sections_order: List[str], text: str) -> Dict[str, Optional[int]]:
    """
    Пытается исправить порядок секций, если они найдены в неправильном порядке.
    
    Args:
        bounds: Словарь с индексами начала каждой секции
        sections_order: Ожидаемый порядок секций
        text: Текст документа
        
    Returns:
        Исправленный словарь с индексами секций
    """
    corrected_bounds = bounds.copy()
    
    # Получаем найденные индексы в правильном порядке
    found_indices = [(section, idx) for section, idx in bounds.items() if idx is not None]
    found_indices.sort(key=lambda x: x[1])
    
    # Если найденных секций меньше, чем ожидается, добавляем отсутствующие
    if len(found_indices) < len(sections_order):
        # Находим отсутствующие секции
        found_sections = [section for section, _ in found_indices]
        missing_sections = [section for section in sections_order if section not in found_sections]
        
        # Для каждой отсутствующей секции пытаемся найти подходящее место
        for section in missing_sections:
            # Определяем, где должна быть секция относительно найденных
            section_index = sections_order.index(section)
            
            # Находим предыдущую и следующую секции
            prev_index = next((bounds[s] for s in sections_order[:section_index] if bounds[s] is not None), None)
            next_index = next((bounds[s] for s in sections_order[section_index+1:] if bounds[s] is not None), None)
            
            # Определяем диапазон для поиска
            search_start = prev_index if prev_index is not None else 0
            search_end = next_index if next_index is not None else len(text)
            
            # Пытаемся найти секцию в этом диапазоне
            search_text = text[search_start:search_end]
            
            # Ищем маркеры этой секции
            for marker in SECTION_MARKERS[section]:
                match = re.search(marker, search_text, re.DOTALL)
                if match:
                    corrected_bounds[section] = search_start + match.start()
                    break
    
    # Проверяем порядок секций после коррекции
    corrected_indices = [idx for idx in [corrected_bounds[section] for section in sections_order] if idx is not None]
    if corrected_indices != sorted(corrected_indices):
        # Если все еще не в порядке, делаем последнюю попытку
        for i, section in enumerate(sections_order):
            if corrected_bounds[section] is None:
                # Если секция не найдена, используем примерное положение
                if i == 0:  # Первая секция
                    corrected_bounds[section] = 0
                elif i == len(sections_order) - 1:  # Последняя секция
                    last_found = max((idx for idx in corrected_bounds.values() if idx is not None), default=0)
                    corrected_bounds[section] = last_found
                else:  # Средняя секция
                    prev_found = max((idx for s, idx in corrected_bounds.items() 
                                    if idx is not None and sections_order.index(s) < i), default=0)
                    next_found = min((idx for s, idx in corrected_bounds.items() 
                                    if idx is not None and sections_order.index(s) > i), default=len(text))
                    corrected_bounds[section] = (prev_found + next_found) // 2
    
    return corrected_bounds

def extract_sections_by_bounds(text: str, bounds: Dict[str, Optional[int]]) -> Dict[str, str]:
    """
    Извлекает текст для каждой секции на основе найденных индексов.
    
    Args:
        text: Текст документа
        bounds: Словарь с индексами начала каждой секции
        
    Returns:
        Словарь с текстами для каждой секции
    """
    sections = {
        'facts': '',
        'reasoning': '',
        'conclusion': ''
    }
    
    # Определяем конец каждой секции на основе начала следующей
    if bounds['facts'] is not None:
        facts_end = bounds['reasoning'] if bounds['reasoning'] is not None else bounds['conclusion']
        if facts_end is not None:
            sections['facts'] = text[bounds['facts']:facts_end].strip()
        else:
            sections['facts'] = text[bounds['facts']:].strip()
    
    if bounds['reasoning'] is not None:
        reasoning_end = bounds['conclusion'] if bounds['conclusion'] is not None else None
        if reasoning_end is not None:
            sections['reasoning'] = text[bounds['reasoning']:reasoning_end].strip()
        else:
            sections['reasoning'] = text[bounds['reasoning']:].strip()
    
    if bounds['conclusion'] is not None:
        sections['conclusion'] = text[bounds['conclusion']:].strip()
    
    return sections

def assess_sections_quality(sections: Dict[str, str]) -> str:
    """
    Оценивает качество разделения текста на секции.
    
    Args:
        sections: Словарь с текстами для каждой секции
        
    Returns:
        Строка с оценкой качества: 'высокое', 'среднее' или 'низкое'
    """
    # Подсчитываем количество заполненных секций
    filled_sections = sum(1 for section in sections.values() if section and len(section) > 100)
    
    # Проверяем соотношение длин секций
    facts_len = len(sections['facts']) if sections['facts'] else 0
    reasoning_len = len(sections['reasoning']) if sections['reasoning'] else 0
    conclusion_len = len(sections['conclusion']) if sections['conclusion'] else 0
    
    total_len = facts_len + reasoning_len + conclusion_len
    
    if total_len == 0:
        return 'низкое'
    
    # Вычисляем процентные соотношения
    facts_ratio = facts_len / total_len if total_len > 0 else 0
    reasoning_ratio = reasoning_len / total_len if total_len > 0 else 0
    conclusion_ratio = conclusion_len / total_len if total_len > 0 else 0
    
    # Расширяем диапазоны типичных соотношений для разных типов судебных документов
    typical_ratios = {
        'facts': (0.10, 0.65),  # Расширяем диапазон
        'reasoning': (0.20, 0.75),  # Расширяем диапазон
        'conclusion': (0.05, 0.45)   # Расширяем диапазон
    }
    
    # Детекция особых случаев в зависимости от типа документа
    # 1. Документ с длинной фактической частью (например, в гражданских делах)
    case_long_facts = facts_len > reasoning_len and facts_ratio < 0.75 and facts_ratio > 0.40
    
    # 2. Документ с длинной мотивировочной частью (например, обзоры практики)
    case_long_reasoning = reasoning_len > facts_len and reasoning_ratio < 0.75 and reasoning_ratio > 0.40
    
    # 3. Документ с короткой заключительной частью (типично для многих документов)
    case_short_conclusion = conclusion_ratio < 0.20 and conclusion_len > 100
    
    # Проверяем, сколько секций попадают в типичные диапазоны
    in_range_count = 0
    if typical_ratios['facts'][0] <= facts_ratio <= typical_ratios['facts'][1]:
        in_range_count += 1
    if typical_ratios['reasoning'][0] <= reasoning_ratio <= typical_ratios['reasoning'][1]:
        in_range_count += 1
    if typical_ratios['conclusion'][0] <= conclusion_ratio <= typical_ratios['conclusion'][1]:
        in_range_count += 1
    
    # Проверяем базовые требования к содержанию секций
    meaningful_content = True
    for section_name, content in sections.items():
        if content:
            # Проверяем, что секция не содержит только пробелы или короткий текст
            if len(content.strip()) < 50:
                meaningful_content = False
                break
            
            # Проверяем, что секция не содержит неправильный контент
            if section_name == 'facts' and ('руководствуясь статьями' in content.lower() or 'постановил' in content.lower()):
                meaningful_content = False
                break
            if section_name == 'conclusion' and ('установил' in content.lower() or 'материалы дела' in content.lower()):
                meaningful_content = False
                break
    
    # Определяем качество на основе всех факторов
    if filled_sections == 3 and meaningful_content:
        if in_range_count == 3:
            return 'высокое'
        elif in_range_count == 2 and (case_long_facts or case_long_reasoning or case_short_conclusion):
            return 'высокое'
        elif in_range_count >= 1:
            return 'среднее'
        else:
            return 'низкое'
    elif filled_sections == 2 and meaningful_content:
        if in_range_count >= 2:
            return 'среднее'
        elif in_range_count == 1 and (case_long_facts or case_long_reasoning):
            return 'среднее'
        else:
            return 'низкое'
    else:
        return 'низкое'

def apply_extended_heuristics(text: str, current_sections: Dict[str, str], doc_type: str) -> Dict[str, str]:
    """
    Применяет расширенные эвристики для разделения текста на секции.
    
    Args:
        text: Текст документа
        current_sections: Текущие найденные секции
        doc_type: Тип документа
        
    Returns:
        Обновленный словарь с секциями
    """
    sections = current_sections.copy()
    
    # Если все секции уже заполнены и их качество приемлемо
    if all(sections.values()) and all(len(section) > 200 for section in sections.values()):
        return sections
    
    # Пытаемся использовать семантический анализ текста для разделения
    try:
        # Ищем семантические части на основе содержания
        semantic_parts = identify_semantic_parts(text)
        
        # Заполняем недостающие секции на основе семантических частей
        for section_name, parts in semantic_parts.items():
            if (not sections[section_name] or len(sections[section_name]) < 200) and parts:
                # Объединяем все найденные части секции
                section_text = ""
                for start, end in parts:
                    section_text += text[start:end] + "\n\n"
                sections[section_name] = section_text.strip()
    except Exception as e:
        logging.warning(f"Ошибка при семантическом анализе: {str(e)}")
    
    # Проверяем отношение длины facts к reasoning
    # Если facts намного длиннее reasoning, возможно часть facts относится к reasoning
    facts_len = len(sections['facts']) if sections['facts'] else 0
    reasoning_len = len(sections['reasoning']) if sections['reasoning'] else 0
    
    if facts_len > reasoning_len * 2 and facts_len > 1000 and reasoning_len > 0:
        # Переносим часть facts в reasoning
        facts_text = sections['facts']
        # Пытаемся найти естественную границу (по абзацам)
        paragraphs = re.split(r'\n\s*\n', facts_text)
        if len(paragraphs) > 2:
            # Находим примерную середину текста фактов
            mid_point = len(facts_text) // 2
            # Ищем ближайшую границу абзаца к середине
            cut_point = 0
            current_pos = 0
            for para in paragraphs:
                next_pos = current_pos + len(para) + 2
                if abs(next_pos - mid_point) < abs(current_pos - mid_point) and current_pos < mid_point:
                    cut_point = next_pos
                current_pos = next_pos
            
            if cut_point > 0:
                # Перераспределяем текст между секциями
                sections['reasoning'] = facts_text[cut_point:] + "\n\n" + sections['reasoning']
                sections['facts'] = facts_text[:cut_point]
    
    # Если все еще есть пустые секции, используем простое пропорциональное разделение
    if not all(sections.values()):
        if not any(sections.values()):
            # Если ни одна секция не найдена, делим текст пропорционально
            text_len = len(text)
            sections['facts'] = text[:int(text_len * 0.3)].strip()
            sections['reasoning'] = text[int(text_len * 0.3):int(text_len * 0.8)].strip()
            sections['conclusion'] = text[int(text_len * 0.8):].strip()
        else:
            # Если найдена хотя бы одна секция, используем ее как опорную точку
            if sections['facts'] and not sections['reasoning'] and not sections['conclusion']:
                text_after_facts = text[text.find(sections['facts']) + len(sections['facts']):].strip()
                text_len = len(text_after_facts)
                sections['reasoning'] = text_after_facts[:int(text_len * 0.7)].strip()
                sections['conclusion'] = text_after_facts[int(text_len * 0.7):].strip()
            
            elif not sections['facts'] and sections['reasoning'] and not sections['conclusion']:
                text_before_reasoning = text[:text.find(sections['reasoning'])].strip()
                text_after_reasoning = text[text.find(sections['reasoning']) + len(sections['reasoning']):].strip()
                sections['facts'] = text_before_reasoning
                sections['conclusion'] = text_after_reasoning
            
            elif not sections['facts'] and not sections['reasoning'] and sections['conclusion']:
                text_before_conclusion = text[:text.find(sections['conclusion'])].strip()
                text_len = len(text_before_conclusion)
                sections['facts'] = text_before_conclusion[:int(text_len * 0.4)].strip()
                sections['reasoning'] = text_before_conclusion[int(text_len * 0.4):].strip()
    
    return sections

def identify_semantic_parts(text: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Идентифицирует семантические части текста на основе содержания.
    
    Args:
        text: Текст документа
        
    Returns:
        Словарь с списками начал и концов каждой семантической части
    """
    parts = {
        'facts': [],
        'reasoning': [],
        'conclusion': []
    }
    
    # Разделяем текст на параграфы
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    # Если параграфов мало, пробуем разделить по предложениям
    if len(paragraphs) < 10:
        try:
            # Используем NLTK для разделения на предложения
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            # Группируем предложения в параграфы (по 3-5 предложений)
            paragraph_size = max(3, min(5, len(sentences) // 10))
            paragraphs = []
            for i in range(0, len(sentences), paragraph_size):
                paragraphs.append(' '.join(sentences[i:i+paragraph_size]))
        except Exception as e:
            logging.warning(f"Ошибка при разделении на предложения: {str(e)}")
    
    # Характерные слова для каждой секции с увеличенным списком
    facts_words = [
        'иск', 'заявление', 'требование', 'установил', 'указал', 'представитель', 
        'обратился', 'отношении', 'факт', 'обстоятельство', 'заявитель', 'материалы дела',
        'согласно материалам', 'истец', 'ответчик', 'сторона', 'документ', 'представленный',
        'приложение', 'поступило', 'обоснование', 'ссылается', 'указывает на', 'договор',
        'соглашение', 'возражение', 'протокол', 'акт', 'жалоба', 'ходатайство',
        # Дополнительные слова для фактов
        'дело', 'слушание', 'рассмотрение', 'запрос', 'инициатива', 'иск предъявлен',
        'подал иск', 'обратился в суд', 'представлены доказательства', 'дата', 'период',
        'компания', 'лицо', 'организация', 'учреждение', 'предприятие', 'сделка',
        'письмо', 'сообщение', 'адрес', 'обращение', 'уведомление', 'претензия'
    ]
    
    reasoning_words = [
        'считает', 'оценивая', 'доказательства', 'доводы', 'согласно', 'статья', 
        'закон', 'мнение', 'следует', 'вывод', 'обоснование', 'основания', 'полагает',
        'приходит к выводу', 'установлено', 'норма', 'правоотношения', 'применение',
        'толкование', 'анализ', 'суд установил', 'суд считает', 'в соответствии',
        'согласно статье', 'таким образом', 'при таких обстоятельствах', 'при этом',
        # Дополнительные слова для мотивировки
        'нормативный акт', 'кодекс', 'федеральный закон', 'постановление', 'разъяснение',
        'пленум', 'правовая позиция', 'судебная практика', 'практика применения',
        'коллегия', 'подлежит', 'не подлежит', 'следовательно', 'из этого следует',
        'вместе с тем', 'однако', 'исходя из', 'учитывая', 'принимая во внимание',
        'в силу', 'позиция суда', 'исследовав', 'проанализировав', 'рассмотрев'
    ]
    
    conclusion_words = [
        'постановил', 'решил', 'определил', 'руководствуясь', 'отказать', 'удовлетворить', 
        'взыскать', 'признать', 'обязать', 'возложить', 'исковые требования', 'руководствуясь статьями',
        'на основании изложенного', 'в соответствии со статьями', 'суд постановляет',
        'суд решает', 'суд определяет', 'с учетом изложенного', 'исходя из изложенного',
        'на основании вышеизложенного', 'в результате рассмотрения', 'по результатам',
        # Дополнительные слова для заключения
        'оставить без изменения', 'отменить', 'изменить', 'направить', 'передать',
        'прекратить производство', 'оставить без рассмотрения', 'апелляционную жалобу',
        'кассационную жалобу', 'жалобу', 'заявление', 'иск', 'отклонить', 'отказать в удовлетворении',
        'постановление окончательное', 'обжалованию не подлежит', 'может быть обжаловано',
        'срок для обжалования', 'вступает в силу', 'немедленному исполнению'
    ]
    
    # Веса для более значимых слов - увеличиваем веса для наиболее характерных фраз
    facts_weights = {
        'установил': 2.0, 'материалы дела': 2.0, 'обстоятельство': 1.8,
        'согласно материалам': 1.8, 'истец': 1.5, 'ответчик': 1.5,
        'заявитель обратился': 2.0, 'истец обратился': 2.0,
        'в обоснование заявленных требований': 2.5,
        'из материалов дела следует': 2.5,
        'в судебном заседании установлено': 2.5,
        'дело': 1.3, 'слушание': 1.3, 'рассмотрение': 1.3
    }
    
    reasoning_weights = {
        'суд считает': 2.5, 'в соответствии': 2.0, 'согласно статье': 2.0,
        'приходит к выводу': 2.0, 'таким образом': 1.8, 'суд установил': 2.0,
        'исследовав материалы дела': 2.5, 'проанализировав доводы сторон': 2.5,
        'суд приходит к выводу': 2.5, 'оценив доказательства': 2.0,
        'обсудив доводы жалобы': 2.0, 'правовая позиция': 2.0,
        'судебная практика': 2.0, 'позиция суда': 2.0
    }
    
    conclusion_weights = {
        'руководствуясь статьями': 2.5, 'суд постановляет': 2.5, 'суд решает': 2.5,
        'на основании изложенного': 2.0, 'исходя из изложенного': 2.0,
        'постановил': 2.5, 'решил': 2.5, 'определил': 2.5,
        'отказать': 1.8, 'удовлетворить': 1.8, 'взыскать': 1.8,
        'признать': 1.8, 'обязать': 1.8, 'возложить': 1.8
    }
    
    # Определяем тип документа для корректировки весов
    doc_type_indicators = {
        'civil': ['гражданское дело', 'истец', 'ответчик', 'иск'],
        'criminal': ['уголовное дело', 'обвиняемый', 'подсудимый'],
        'administrative': ['административное дело', 'административное правонарушение'],
        'arbitration': ['арбитражный суд', 'экономический спор'],
        'bulletin': ['бюллетень', 'обзор судебной практики']
    }
    
    doc_type = 'unknown'
    for t, indicators in doc_type_indicators.items():
        for indicator in indicators:
            if indicator in text.lower():
                doc_type = t
                break
        if doc_type != 'unknown':
            break
    
    # Корректируем веса в зависимости от типа документа
    if doc_type == 'bulletin' or doc_type == 'review':
        # В бюллетенях важнее разделы с решениями и выводами
        for key in reasoning_weights:
            reasoning_weights[key] *= 1.3
        for key in conclusion_weights:
            conclusion_weights[key] *= 1.2
    
    # Используем расширенную функцию для вычисления весов слов в тексте
    def calculate_weighted_score(text: str, words: List[str], weights: Optional[Dict[str, float]] = None) -> float:
        text_lower = text.lower()
        score = 0
        
        # Прямой подсчет вхождений слов с учетом весов
        for word in words:
            weight = weights.get(word, 1.0) if weights else 1.0
            
            # Проверяем точное вхождение слова
            count = text_lower.count(word.lower())
            score += count * weight
            
            # Дополнительные баллы за вхождение в начало параграфа
            if word.lower() in text_lower[:min(100, len(text_lower))]:
                score += weight * 0.5
            
            # Дополнительные баллы для составных фраз
            if len(word.split()) > 1 and word.lower() in text_lower:
                score += weight * 0.7  # Бонус за точное совпадение фразы
        
        # Нормализация для длинных текстов
        if len(text) > 500:
            score = score * (500 / len(text)) * 1.5
            
        return score
    
    current_pos = 0
    for para_index, para in enumerate(paragraphs):
        para_len = len(para)
        
        # Подсчитываем взвешенные баллы для каждой секции
        facts_score = calculate_weighted_score(para, facts_words, facts_weights)
        reasoning_score = calculate_weighted_score(para, reasoning_words, reasoning_weights)
        conclusion_score = calculate_weighted_score(para, conclusion_words, conclusion_weights)
        
        # Учитываем позицию параграфа в документе с более гибким подходом
        doc_position = para_index / len(paragraphs)
        
        # Корректируем баллы в зависимости от позиции
        if doc_position < 0.3:  # Первая треть документа
            facts_score *= 1.3
            conclusion_score *= 0.7
        elif doc_position > 0.7:  # Последняя треть документа
            conclusion_score *= 1.3
            facts_score *= 0.7
        else:  # Средняя треть документа
            reasoning_score *= 1.3
            
        # Корректировка для специальных типов документов
        if doc_type == 'bulletin' or doc_type == 'review':
            # В бюллетенях важнее содержание, чем позиция
            if facts_score > 3.0:
                facts_score *= 1.2
            if reasoning_score > 3.0:
                reasoning_score *= 1.2
            if conclusion_score > 3.0:
                conclusion_score *= 1.2
        
        # Определяем, к какой секции отнести параграф
        scores = {
            'facts': facts_score,
            'reasoning': reasoning_score,
            'conclusion': conclusion_score
        }
        
        # Проверяем, что есть значимый балл
        max_score = max(scores.values())
        if max_score > 0.3:  # Снижаем порог для отнесения к секции
            max_section = max(scores.items(), key=lambda x: x[1])[0]
            parts[max_section].append((current_pos, current_pos + para_len))
        
        current_pos += para_len + 2  # +2 для учета разделителей параграфов
    
    # Объединяем близко расположенные части одной секции
    for section in parts:
        if len(parts[section]) > 1:
            merged_parts = []
            if parts[section]:
                current_start, current_end = parts[section][0]
                
                for start, end in parts[section][1:]:
                    if start - current_end < 1000:  # Увеличиваем расстояние для объединения
                        current_end = end  # Объединяем с предыдущей частью
                    else:
                        merged_parts.append((current_start, current_end))
                        current_start, current_end = start, end
                
                merged_parts.append((current_start, current_end))
                parts[section] = merged_parts
    
    # Если какая-то секция не найдена, пытаемся разделить текст эвристически
    for section in parts:
        if not parts[section] and text:
            if section == 'facts':
                # Факты обычно в начале
                start = 0
                end = min(len(text) // 3, 3000)
                parts[section].append((start, end))
            elif section == 'reasoning':
                # Мотивировка обычно в середине
                start = len(text) // 3
                end = 2 * len(text) // 3
                parts[section].append((start, end))
            elif section == 'conclusion':
                # Заключение обычно в конце
                start = 2 * len(text) // 3
                end = len(text)
                parts[section].append((start, end))
    
    return parts

def balance_section_lengths(sections: Dict[str, str], full_text: str) -> Dict[str, str]:
    """
    Балансирует длины секций в соответствии с ожидаемыми пропорциями.
    
    Args:
        sections: Словарь с текстами для каждой секции
        full_text: Полный текст документа
        
    Returns:
        Сбалансированный словарь с секциями
    """
    balanced_sections = sections.copy()
    
    facts_len = len(sections['facts']) if sections['facts'] else 0
    reasoning_len = len(sections['reasoning']) if sections['reasoning'] else 0
    conclusion_len = len(sections['conclusion']) if sections['conclusion'] else 0
    
    # Если секции не заполнены, нечего балансировать
    if not any(sections.values()):
        return sections
    
    # Проверяем все возможные проблемы с балансом секций
    
    # 1. Проблема: факты длиннее мотивировки и возможно содержат часть мотивировки
    if facts_len > reasoning_len * 1.5 and facts_len > 1000 and reasoning_len > 0:
        try:
            # Пытаемся найти переходную часть от фактов к мотивировке
            facts_text = sections['facts']
            
            # Ищем ключевые слова мотивировочной части в фактах
            reasoning_markers = [
                'считает', 'суд полагает', 'в соответствии', 'согласно', 'статьей', 
                'законом', 'применение', 'толкование', 'правовая позиция',
                'исходя из', 'учитывая', 'принимая во внимание', 'оценивая',
                'таким образом', 'следовательно', 'суд приходит к выводу',
                'в силу', 'согласно положениям', 'при применении'
            ]
            
            # Разбиваем текст фактов на параграфы
            paragraphs = re.split(r'\n\s*\n', facts_text)
            if len(paragraphs) > 2:
                # Ищем параграфы, которые больше похожи на мотивировку
                reasoning_paragraphs = []
                facts_paragraphs = []
                
                for i, para in enumerate(paragraphs):
                    # Проверяем, сколько маркеров мотивировки содержится в параграфе
                    marker_count = sum(1 for marker in reasoning_markers if marker in para.lower())
                    
                    # Если это последний параграф или один из последних, и он содержит маркеры мотивировки
                    if marker_count > 0 and (i > len(paragraphs) * 0.5 or i >= len(paragraphs) - 3):
                        reasoning_paragraphs.append(para)
                    else:
                        facts_paragraphs.append(para)
                
                # Если нашли параграфы с признаками мотивировки
                if reasoning_paragraphs:
                    # Обновляем секции
                    balanced_sections['facts'] = '\n\n'.join(facts_paragraphs)
                    balanced_sections['reasoning'] = '\n\n'.join(reasoning_paragraphs) + '\n\n' + sections['reasoning']
        except Exception as e:
            logging.warning(f"Ошибка при балансировке секций фактов и мотивировки: {str(e)}")
    
    # 2. Проблема: мотивировка длиннее фактов и возможно содержит часть заключения
    if reasoning_len > facts_len * 3 and reasoning_len > 3000 and conclusion_len < 500:
        try:
            # Пытаемся найти переходную часть от мотивировки к заключению
            reasoning_text = sections['reasoning']
            
            # Ищем ключевые слова заключительной части в мотивировке
            conclusion_markers = [
                'руководствуясь', 'постановил', 'решил', 'определил', 'отказать', 
                'удовлетворить', 'взыскать', 'признать', 'обязать', 'возложить',
                'прекратить', 'на основании изложенного', 'в соответствии со статьями',
                'исходя из изложенного', 'с учетом изложенного'
            ]
            
            # Разбиваем текст мотивировки на параграфы
            paragraphs = re.split(r'\n\s*\n', reasoning_text)
            if len(paragraphs) > 2:
                # Ищем параграфы, которые больше похожи на заключение
                conclusion_paragraphs = []
                reasoning_paragraphs = []
                
                for i, para in enumerate(paragraphs):
                    # Проверяем, сколько маркеров заключения содержится в параграфе
                    marker_count = sum(1 for marker in conclusion_markers if marker in para.lower())
                    
                    # Если это последний параграф или один из последних, и он содержит маркеры заключения
                    if marker_count > 0 and (i > len(paragraphs) * 0.7 or i >= len(paragraphs) - 2):
                        conclusion_paragraphs.append(para)
                    else:
                        reasoning_paragraphs.append(para)
                
                # Если нашли параграфы с признаками заключения
                if conclusion_paragraphs:
                    # Обновляем секции
                    balanced_sections['reasoning'] = '\n\n'.join(reasoning_paragraphs)
                    if sections['conclusion']:
                        balanced_sections['conclusion'] = '\n\n'.join(conclusion_paragraphs) + '\n\n' + sections['conclusion']
                    else:
                        balanced_sections['conclusion'] = '\n\n'.join(conclusion_paragraphs)
        except Exception as e:
            logging.warning(f"Ошибка при балансировке секций мотивировки и заключения: {str(e)}")
    
    # 3. Проблема: очень короткое или отсутствующее заключение
    if (not sections['conclusion'] or len(sections['conclusion']) < 100) and len(full_text) > 1000:
        try:
            # Ищем в конце полного текста возможное заключение
            last_part = full_text[-min(1000, len(full_text)//3):]
            
            # Ищем ключевые маркеры заключения
            conclusion_markers = [
                'руководствуясь', 'постановил', 'решил', 'определил', 'отказать', 
                'удовлетворить', 'взыскать', 'признать', 'на основании изложенного',
                'исходя из изложенного', 'с учетом изложенного'
            ]
            
            for marker in conclusion_markers:
                match_pos = last_part.lower().find(marker)
                if match_pos >= 0:
                    # Нашли маркер заключения, извлекаем текст от него до конца
                    conclusion_text = last_part[match_pos:]
                    
                    # Если уже есть короткое заключение, объединяем их
                    if sections['conclusion']:
                        balanced_sections['conclusion'] = sections['conclusion'] + '\n\n' + conclusion_text
                    else:
                        balanced_sections['conclusion'] = conclusion_text
                    
                    break
        except Exception as e:
            logging.warning(f"Ошибка при поиске заключения в конце текста: {str(e)}")
    
    # 4. Проверяем общий баланс секций после всех корректировок
    balanced_facts_len = len(balanced_sections['facts']) if balanced_sections['facts'] else 0
    balanced_reasoning_len = len(balanced_sections['reasoning']) if balanced_sections['reasoning'] else 0
    balanced_conclusion_len = len(balanced_sections['conclusion']) if balanced_sections['conclusion'] else 0
    
    total_len = balanced_facts_len + balanced_reasoning_len + balanced_conclusion_len
    
    # Если все еще не хватает какой-то секции, пытаемся создать ее из текста
    if total_len > 0:
        if not balanced_sections['facts']:
            # Если нет фактов, берем начало текста
            balanced_sections['facts'] = full_text[:min(len(full_text)//3, 2000)]
        
        if not balanced_sections['reasoning']:
            # Если нет мотивировки, берем середину текста
            start = len(full_text)//3
            end = min(2*len(full_text)//3, len(full_text) - 500)
            balanced_sections['reasoning'] = full_text[start:end]
        
        if not balanced_sections['conclusion']:
            # Если нет заключения, берем конец текста
            balanced_sections['conclusion'] = full_text[max(2*len(full_text)//3, len(full_text) - 1000):]
    
    return balanced_sections

def apply_nlp_heuristics(text: str, current_sections: Dict[str, str]) -> Dict[str, str]:
    """
    Применяет продвинутые эвристики на основе NLP для разделения текста.
    
    Args:
        text: Полный текст документа
        current_sections: Текущие найденные секции
        
    Returns:
        Обновленный словарь с секциями
    """
    sections = current_sections.copy()
    
    try:
        # Проверяем, заполнены ли уже все секции с достаточным объемом
        if all(len(section) > 300 for section in sections.values() if section):
            # Если все секции уже заполнены достаточным объемом, применяем только балансировку
            if all(sections.values()):
                return balance_section_lengths(sections, text)
        
        # Разбиваем текст на предложения
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            # Если не удалось использовать NLTK, используем простое разделение
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Если предложений слишком мало, возвращаем текущие секции
        if len(sentences) < 5:
            return sections
        
        # Готовим словари для характерных слов
        facts_words = set([
            'иск', 'заявление', 'требование', 'установил', 'указал', 'представитель', 
            'обратился', 'отношении', 'факт', 'обстоятельство', 'заявитель', 'материалы дела',
            'истец', 'ответчик', 'договор', 'соглашение', 'дата', 'период', 'компания', 
            'обязательство', 'уведомление', 'претензия', 'запрос', 'обращение', 'дело',
            'слушание', 'рассмотрение', 'суд', 'инстанция', 'жалоба', 'документ'
        ])
        
        reasoning_words = set([
            'считает', 'оценивая', 'доказательства', 'доводы', 'согласно', 'статья', 
            'закон', 'мнение', 'следует', 'вывод', 'обоснование', 'основания', 'полагает',
            'приходит к выводу', 'установлено', 'норма', 'правоотношения', 'применение',
            'толкование', 'анализ', 'нормативный', 'положение', 'регулирование', 
            'правовая позиция', 'практика', 'суд установил', 'суд считает', 'кодекс',
            'постановление', 'пленум', 'определение', 'коллегия', 'президиум'
        ])
        
        conclusion_words = set([
            'постановил', 'решил', 'определил', 'руководствуясь', 'отказать', 'удовлетворить', 
            'взыскать', 'признать', 'обязать', 'возложить', 'исковые требования', 'руководствуясь статьями',
            'прекратить', 'отменить', 'утвердить', 'оставить', 'без изменения', 'в удовлетворении отказать',
            'в иске отказать', 'иск удовлетворить', 'передать', 'апелляционную жалобу', 'кассационную жалобу',
            'вступает в силу', 'обжалованию подлежит', 'обжалованию не подлежит'
        ])
        
        # Классифицируем каждое предложение с улучшенной логикой
        sentence_types = []
        for i, sentence in enumerate(sentences):
            lower_sentence = sentence.lower()
            
            # Считаем количество характерных слов в предложении с учетом частичных совпадений
            facts_score = sum(1 for word in facts_words if word.lower() in lower_sentence)
            reasoning_score = sum(1 for word in reasoning_words if word.lower() in lower_sentence)
            conclusion_score = sum(1 for word in conclusion_words if word.lower() in lower_sentence)
            
            # Учитываем позицию предложения в документе
            position_factor = i / len(sentences)
            if position_factor < 0.3:
                facts_score *= 1.5
            elif position_factor > 0.7:
                conclusion_score *= 1.5
            else:
                reasoning_score *= 1.3
            
            # Определяем тип предложения
            max_score = max(facts_score, reasoning_score, conclusion_score)
            if max_score == 0:
                # Если нет характерных слов, классифицируем по позиции в тексте
                if position_factor < 0.3:
                    sentence_types.append('facts')
                elif position_factor > 0.7:
                    sentence_types.append('conclusion')
                else:
                    sentence_types.append('reasoning')
            elif facts_score == max_score:
                sentence_types.append('facts')
            elif reasoning_score == max_score:
                sentence_types.append('reasoning')
            else:
                sentence_types.append('conclusion')
        
        # Объединяем последовательные предложения одного типа в блоки
        blocks = []
        current_type = sentence_types[0]
        current_block = [sentences[0]]
        
        for i in range(1, len(sentences)):
            if sentence_types[i] == current_type:
                current_block.append(sentences[i])
            else:
                blocks.append((current_type, ' '.join(current_block)))
                current_type = sentence_types[i]
                current_block = [sentences[i]]
        
        # Добавляем последний блок
        blocks.append((current_type, ' '.join(current_block)))
        
        # Объединяем блоки каждого типа
        section_texts = {'facts': [], 'reasoning': [], 'conclusion': []}
        for block_type, block_text in blocks:
            section_texts[block_type].append(block_text)
        
        # Заполняем секции
        for section_type, texts in section_texts.items():
            if texts and (not sections[section_type] or len(sections[section_type]) < 200):
                sections[section_type] = '\n\n'.join(texts)
        
        # Применяем дополнительную балансировку
        return balance_section_lengths(sections, text)
        
    except Exception as e:
        logging.warning(f"Ошибка при применении NLP эвристик: {str(e)}")
        return sections

def process_file(input_file: str, output_file: str) -> None:
    """
    Обрабатывает отдельный файл, разделяя его на секции.
    
    Args:
        input_file: Путь к входному файлу
        output_file: Путь к выходному файлу
    """
    try:
        logging.info(f"Обработка файла {input_file}")
        
        # Чтение входного файла
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Если это JSON файл, пытаемся прочитать как JSON
        if input_file.endswith('.json'):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and 'text' in data:
                    text = data['text']
            except json.JSONDecodeError:
                logging.warning(f"Файл {input_file} имеет расширение .json, но не является валидным JSON.")
        
        # Разделение текста на секции
        sections = split_into_sections(text)
        
        # Оценка качества разделения
        quality = assess_sections_quality(sections)
        
        # Формирование результата
        result = {
            "filename": os.path.basename(input_file),
            "sections": sections,
            "quality": quality
        }
        
        # Запись результата в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Файл {input_file} обработан с качеством: {quality}")
        return quality
        
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {input_file}: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Запись информации об ошибке
        error_result = {
            "filename": os.path.basename(input_file),
            "error": str(e)
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
        except Exception as write_error:
            logging.error(f"Ошибка при записи информации об ошибке: {str(write_error)}")
        
        return "ошибка"

def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Обрабатывает все файлы в директории, разделяя их на секции.
    
    Args:
        input_dir: Путь к входной директории
        output_dir: Путь к выходной директории
    """
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Находим все текстовые файлы в директории
    input_files = []
    for ext in ['.txt', '.json']:
        input_files.extend(list(pathlib.Path(input_dir).glob(f'**/*{ext}')))
    
    if not input_files:
        logging.warning(f"В директории {input_dir} не найдено текстовых файлов.")
        return
    
    logging.info(f"Найдено {len(input_files)} файлов для обработки.")
    
    # Статистика качества
    quality_stats = Counter()
    
    # Обрабатываем каждый файл
    for input_file in tqdm(input_files, desc="Обработка файлов"):
        rel_path = input_file.relative_to(input_dir)
        output_file = os.path.join(output_dir, rel_path.with_suffix('.json'))
        
        # Создаем необходимые поддиректории
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Обрабатываем файл
        quality = process_file(str(input_file), output_file)
        quality_stats[quality] += 1
    
    # Создаем файл с общей информацией о результатах
    summary = {
        "total_documents": len(input_files),
        "quality_stats": dict(quality_stats),
        "documents": []
    }
    
    # Добавляем информацию о каждом документе
    for input_file in input_files:
        rel_path = input_file.relative_to(input_dir)
        output_file = os.path.join(output_dir, rel_path.with_suffix('.json'))
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                doc_info = json.load(f)
            
            # Добавляем только базовую информацию
            if "error" in doc_info:
                summary["documents"].append({
                    "filename": doc_info["filename"],
                    "error": doc_info["error"]
                })
            else:
                summary["documents"].append({
                    "filename": doc_info["filename"],
                    "quality": doc_info["quality"]
                })
        except Exception as e:
            summary["documents"].append({
                "filename": os.path.basename(str(input_file)),
                "error": f"Ошибка при чтении обработанного файла: {str(e)}"
            })
    
    # Записываем общую информацию
    summary_file = os.path.join(output_dir, '..', 'analyzed', 'summary.json')
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Обработка завершена. Статистика качества: {dict(quality_stats)}")

def main() -> None:
    """
    Основная функция для запуска скрипта.
    """
    parser = argparse.ArgumentParser(description='Разделение судебных решений на логические секции.')
    parser.add_argument('--input', required=True, help='Путь к входному файлу или директории')
    parser.add_argument('--output', required=True, help='Путь к выходному файлу или директории')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    if os.path.isdir(input_path):
        # Обработка директории
        process_directory(input_path, output_path)
    else:
        # Обработка отдельного файла
        process_file(input_path, output_path)

if __name__ == "__main__":
    main() 