import os
import pathlib
import logging
import re
import pymupdf

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Словарь соответствия латинских символов с диакритикой русским символам
LATIN_TO_CYRILLIC = {
    # Заглавные буквы
    'À': 'А', 'Á': 'А', 'Â': 'В', 'Ã': 'Г', 'Ä': 'Д', 'Å': 'Е',
    'Æ': 'Ж', 'Ç': 'З',
    'È': 'И', 'É': 'Й', 'Ê': 'К', 'Ë': 'Л',
    'Ì': 'М', 'Í': 'Н', 'Î': 'О', 'Ï': 'П',
    'Ð': 'Р', 'Ñ': 'С', 'Ò': 'Т', 'Ó': 'У', 'Ô': 'Ф', 'Õ': 'Х', 'Ö': 'Ц',
    '×': 'Ч', 'Ø': 'Ш', 'Ù': 'Щ', 'Ú': 'Ъ', 'Û': 'Ы', 'Ü': 'Ь',
    'Ý': 'Э', 'Þ': 'Ю', 'ß': 'Я',
    
    # Строчные буквы
    'à': 'а', 'á': 'б', 'â': 'в', 'ã': 'г', 'ä': 'д', 'å': 'е',
    'æ': 'ж', 'ç': 'з',
    'è': 'и', 'é': 'й', 'ê': 'к', 'ë': 'л',
    'ì': 'м', 'í': 'н', 'î': 'о', 'ï': 'п',
    'ð': 'р', 'ñ': 'с', 'ò': 'т', 'ó': 'у', 'ô': 'ф', 'õ': 'х', 'ö': 'ц',
    '÷': 'ч', 'ø': 'ш', 'ù': 'щ', 'ú': 'ъ', 'û': 'ы', 'ü': 'ь',
    'ý': 'э', 'þ': 'ю', 'ÿ': 'я',
    
    # Дополнительные соответствия для часто встречающихся ошибок
    'œ': 'ое', 'º': 'о', '©': 'с', '®': 'р', '«': '"', '»': '"',
    '№': '№', '°': 'о', '±': '+', '¶': 'п', '¬': '-', '¦': '|'
}

def fix_cyrillic_encoding(text):
    """
    Заменяет латинские символы с диакритикой на русские символы.
    
    Args:
        text: Исходный текст с проблемами кодировки
    
    Returns:
        Исправленный текст с корректной кодировкой русских символов
    """
    # Замена символов по словарю
    for latin, cyrillic in LATIN_TO_CYRILLIC.items():
        text = text.replace(latin, cyrillic)
    
    # Дополнительные правила и исправления
    # Исправление "иб" на "об" (часто встречающаяся ошибка)
    text = re.sub(r'иа', 'об', text)
    
    # Исправление "ий" на "ой" в окончаниях
    text = re.sub(r'ий\b', 'ой', text)
    text = re.sub(r'иги\b', 'ого', text)
    text = re.sub(r'ими\b', 'ого', text)
    
    # Исправление предлогов
    text = re.sub(r'\bи\s', 'о ', text)
    text = re.sub(r'\bит\b', 'от', text)
    
    # Исправления для "Российской Федерации"
    text = re.sub(r'Риссийский Фадарации', 'Российской Федерации', text)
    text = re.sub(r'[Рр]иссийск([а-я]{2,})', r'Российск\1', text)
    text = re.sub(r'[Фф]адара([а-я]{2,})', r'Федера\1', text)
    
    # Часто встречающиеся слова
    text = re.sub(r'сод([а-я]{1,})\b', r'суд\1', text)
    text = re.sub(r'вархивн([а-я]{1,})\b', r'верховн\1', text)
    text = re.sub(r'гисодарств([а-я]{1,})\b', r'государств\1', text)
    text = re.sub(r'иаяза([а-я]{1,})\b', r'обяза\1', text)
    text = re.sub(r'закин([а-я]{0,})\b', r'закон\1', text)
    text = re.sub(r'кидекс([а-я]{0,})\b', r'кодекс\1', text)
    text = re.sub(r'спир([а-я]{1,})\b', r'спор\1', text)
    text = re.sub(r'трод([а-я]{1,})\b', r'труд\1', text)
    text = re.sub(r'права([а-я]{1,})\b', r'право\1', text)
    text = re.sub(r'инистранн([а-я]{1,})\b', r'иностранн\1', text)
    
    return text

def convert_pdf_to_text(input_pdf_path: str, output_txt_path: str) -> bool:
    """
    Читает PDF по пути input_pdf_path и сохраняет его текст в output_txt_path.
    
    Args:
        input_pdf_path: Путь к исходному PDF-файлу
        output_txt_path: Путь для сохранения извлеченного текста
    
    Returns:
        True в случае успеха, False в случае ошибки
    """
    try:
        logging.info(f"Начинаем конвертацию файла {input_pdf_path} → {output_txt_path}")
        
        # Определяем, является ли файл бюллетенем или юридическим документом
        is_bulletin = 'бюллетень' in input_pdf_path.lower()
        
        # Открываем PDF-документ с помощью PyMuPDF
        with pymupdf.open(input_pdf_path) as doc:
            text_blocks = []
            
            # Извлекаем текст из каждой страницы
            for page_num, page in enumerate(doc):
                try:
                    # Для бюллетеней применяем специальные параметры извлечения текста
                    if is_bulletin:
                        # Пробуем разные параметры извлечения текста для бюллетеней
                        page_text = page.get_text(
                            flags=pymupdf.TEXT_PRESERVE_LIGATURES | 
                                  pymupdf.TEXT_PRESERVE_WHITESPACE | 
                                  pymupdf.TEXT_PRESERVE_IMAGES |
                                  pymupdf.TEXT_DEHYPHENATE
                        )
                    else:
                        # Стандартное извлечение текста
                        page_text = page.get_text()
                    
                    text_blocks.append(page_text)
                except Exception as e:
                    logging.warning(f"Ошибка при извлечении текста со страницы {page_num+1}: {str(e)}")
            
            # Объединяем текст всех страниц с разделителем
            text = chr(12).join(text_blocks)  # Form feed as page separator
            
            # Для бюллетеней применяем специальную обработку
            if is_bulletin:
                # Нормализуем пробелы и переносы строк
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r' {2,}', '\n', text)
                
                # Исправляем кодировку русских символов
                text = fix_cyrillic_encoding(text)
                
                # Улучшаем форматирование
                text = re.sub(r'([.!?]) ([А-ЯA-Z])', r'\1\n\2', text)  # Новая строка после точки и заглавной буквы
                text = re.sub(r'\n{3,}', '\n\n', text)  # Нормализация многократных переносов
            
            # Сохраняем текст в выходной файл в кодировке UTF-8
            pathlib.Path(output_txt_path).write_text(text, encoding='utf-8')
            
        logging.info(f"Файл {output_txt_path} успешно создан")
        return True
    
    except Exception as e:
        logging.error(f"Ошибка обработки {input_pdf_path}: {str(e)}")
        return False

def process_all_pdfs(input_dir: str, output_dir: str) -> None:
    """
    Конвертирует все PDF-файлы из указанной директории в текстовые файлы.
    
    Args:
        input_dir: Директория с исходными PDF-файлами
        output_dir: Директория для сохранения извлеченных текстов
    """
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список всех PDF-файлов
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    logging.info(f"Всего найдено {len(pdf_files)} PDF-файлов для обработки")
    
    # Обрабатываем каждый PDF-файл
    success_count = 0
    for filename in pdf_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename[:-4] + ".txt")
        
        if convert_pdf_to_text(input_path, output_path):
            success_count += 1
    
    logging.info(f"Обработка завершена. Успешно обработано {success_count} из {len(pdf_files)} файлов")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Конвертация PDF-файлов в текст')
    parser.add_argument('--input-dir', type=str, default='data/raw', 
                        help='Директория с исходными PDF-файлами')
    parser.add_argument('--output-dir', type=str, default='data/processed', 
                        help='Директория для сохранения извлеченных текстов')
    
    args = parser.parse_args()
    
    process_all_pdfs(args.input_dir, args.output_dir) 