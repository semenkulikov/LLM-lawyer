import os
import pathlib
import re
import pymupdf
from loguru import logger

# Расширенный словарь соответствия латинских символов с диакритикой русским символам
# Этот словарь составлен на основе наблюдений за реальными заменами в PDF-файлах
LATIN_TO_CYRILLIC = {
    # Заглавные буквы
    'À': 'А', 'Á': 'Б', 'Â': 'В', 'Ã': 'Г', 'Ä': 'Д', 'Å': 'Е',
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
    
    # Дополнительные символы, часто встречающиеся в PDF
    'º': 'о', '¹': 'и', '²': 'е', '³': 'а',
    '°': 'о', '±': 'и',
    '¼': 'ч', '½': 'ш', '¾': 'ь',
    'µ': 'м', '¿': 'я',
    '®': 'р', '©': 'с',
    '¢': 'ц', '£': 'л',
    '¤': 'д', '¥': 'э',
    '¦': 'и', '§': 'п',
    '¨': 'е', '«': 'к',
    '¬': 'н', '­': 'т',
    '¯': 'в', '·': 'б',
    '»': 'ю', 'ª': 'ж',
    
    # Числа, которые могут быть заменены
    '¡': '1', '¢': '2', '£': '3', '¤': '4', '¥': '5',
    '¦': '6', '§': '7', '¨': '8', '©': '9', 'ª': '0',
    
    # Другие символы
    'ё': 'ё', 'Ё': 'Ё',  # Ё часто сохраняется как есть
    '–': '-', '—': '-',  # Различные тире
    '№': '№',            # Знак номера
    '…': '...',          # Многоточие
    
    # Дополнительные символы с диакритикой
    'ă': 'а', 'ą': 'а', 'ć': 'с', 'č': 'ч', 'ď': 'д', 
    'ē': 'е', 'ĕ': 'е', 'ė': 'е', 'ę': 'е', 'ě': 'е',
    'ğ': 'г', 'ģ': 'г', 'ī': 'и', 'ĭ': 'и', 'į': 'и',
    'ķ': 'к', 'ĺ': 'л', 'ļ': 'л', 'ľ': 'л', 'ł': 'л',
    'ń': 'н', 'ņ': 'н', 'ň': 'н', 'ō': 'о', 'ŏ': 'о',
    'ő': 'о', 'œ': 'э', 'ŕ': 'р', 'ř': 'р', 'ś': 'с',
    'ş': 'с', 'š': 'ш', 'ţ': 'т', 'ť': 'т', 'ū': 'у',
    'ŭ': 'у', 'ů': 'у', 'ű': 'у', 'ų': 'у', 'ź': 'з',
    'ż': 'ж', 'ž': 'ж',
    
    # Заглавные буквы с диакритикой
    'Ă': 'А', 'Ą': 'А', 'Ć': 'С', 'Č': 'Ч', 'Ď': 'Д', 
    'Ē': 'Е', 'Ĕ': 'Е', 'Ė': 'Е', 'Ę': 'Е', 'Ě': 'Е',
    'Ğ': 'Г', 'Ģ': 'Г', 'Ī': 'И', 'Ĭ': 'И', 'Į': 'И',
    'Ķ': 'К', 'Ĺ': 'Л', 'Ļ': 'Л', 'Ľ': 'Л', 'Ł': 'Л',
    'Ń': 'Н', 'Ņ': 'Н', 'Ň': 'Н', 'Ō': 'О', 'Ŏ': 'О',
    'Ő': 'О', 'Œ': 'Э', 'Ŕ': 'Р', 'Ř': 'Р', 'Ś': 'С',
    'Ş': 'С', 'Š': 'Ш', 'Ţ': 'Т', 'Ť': 'Т', 'Ū': 'У',
    'Ŭ': 'У', 'Ů': 'У', 'Ű': 'У', 'Ų': 'У', 'Ź': 'З',
    'Ż': 'Ж', 'Ž': 'Ж',
    
    # Специфические случаи для бюллетеней
    'o': 'о', 'O': 'О',  # Латинские o/O часто используются вместо кириллических
    'e': 'е', 'E': 'Е',  # Латинские e/E часто используются вместо кириллических
    'p': 'р', 'P': 'Р',  # Латинские p/P часто используются вместо кириллических
    'a': 'а', 'A': 'А',  # Латинские a/A часто используются вместо кириллических
    'c': 'с', 'C': 'С',  # Латинские c/C часто используются вместо кириллических
    'x': 'х', 'X': 'Х',  # Латинские x/X часто используются вместо кириллических
    'y': 'у', 'Y': 'У',  # Латинские y/Y часто используются вместо кириллических
    'B': 'В',            # Латинская B часто используется вместо кириллической В
    'H': 'Н',            # Латинская H часто используется вместо кириллической Н
    'K': 'К',            # Латинская K часто используется вместо кириллической К
    'M': 'М',            # Латинская M часто используется вместо кириллической М
    'T': 'Т',            # Латинская T часто используется вместо кириллической Т
}

# Функция для замены латинских символов на кириллические
def replace_latin_with_cyrillic(text):
    """
    Заменяет латинские символы с диакритикой на соответствующие русские символы.
    """
    # Создаем шаблон регулярного выражения для всех ключей словаря
    pattern = re.compile('|'.join(map(re.escape, LATIN_TO_CYRILLIC.keys())))
    
    # Заменяем все найденные символы
    return pattern.sub(lambda m: LATIN_TO_CYRILLIC[m.group(0)], text)

# Исправление типичных ошибок в тексте после конвертации
def fix_common_errors(text):
    """
    Исправляет типичные ошибки в тексте после конвертации из PDF.
    """
    # Замена повторяющихся символов (например, "оо" -> "о")
    text = re.sub(r'([а-яА-ЯёЁ])\1{2,}', r'\1\1', text)
    
    # Исправление часто встречающихся ошибок в словах
    replacements = [
        (r'\bисзвестия\b', 'известия'),
        (r'\bисбранные\b', 'избранные'),
        (r'\bраздиление\b', 'разделение'),
        (r'\bиспользиевание\b', 'использование'),
        (r'\bиаластниги\b', 'областного'),
        (r'\bпрадсадаталь\b', 'председатель'),
        (r'\bвыстопила\b', 'выступила'),
        (r'\bдикладим\b', 'докладом'),
        # Можно добавить другие типичные ошибки
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    return text

def convert_pdf_to_text(input_pdf_path: str, output_txt_path: str) -> None:
    """
    Читает PDF по пути input_pdf_path и сохраняет его текст в output_txt_path.
    
    Args:
        input_pdf_path: Путь к исходному PDF-файлу
        output_txt_path: Путь для сохранения извлеченного текста
    """
    try:
        logger.info(f"Начинаем конвертацию файла {input_pdf_path} → {output_txt_path}")
        
        # Проверяем, является ли файл бюллетенем
        is_bulletin = 'бюллетень' in input_pdf_path.lower()
        
        # Открываем PDF-документ с помощью PyMuPDF
        with pymupdf.open(input_pdf_path) as doc:
            text_pages = []
            
            # Извлекаем текст из всех страниц документа
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                if is_bulletin:
                    # Для бюллетеней используем специальные параметры извлечения текста
                    text = page.get_text("text", flags=pymupdf.TEXT_PRESERVE_LIGATURES | 
                                                 pymupdf.TEXT_PRESERVE_WHITESPACE | 
                                                 pymupdf.TEXT_DEHYPHENATE |
                                                 pymupdf.TEXT_PRESERVE_SPANS)
                    
                    # Применяем функцию замены латинских символов на кириллические
                    text = replace_latin_with_cyrillic(text)
                    
                    # Исправляем типичные ошибки в тексте
                    text = fix_common_errors(text)
                    
                    # Дополнительная обработка для бюллетеней
                    # Удаляем повторяющиеся пробелы и лишние переносы строк
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    
                    # Восстанавливаем абзацы, объединяя перенесенные строки
                    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
                else:
                    # Для обычных документов используем стандартные параметры
                    text = page.get_text()
                
                text_pages.append(text)
            
            # Объединяем текст всех страниц
            full_text = "\n".join(text_pages)
            
            # Создаем директорию для выходного файла, если она не существует
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
            
            # Записываем извлеченный текст в файл
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            logger.info(f"Файл {output_txt_path} успешно создан")
            
    except Exception as e:
        logger.error(f"Ошибка при конвертации файла {input_pdf_path}: {str(e)}")

def process_all_pdfs(input_dir: str, output_dir: str) -> None:
    """
    Обрабатывает все PDF-файлы в указанной директории с проверкой уже обработанных.
    
    Args:
        input_dir: Директория с исходными PDF-файлами
        output_dir: Директория для сохранения извлеченных текстов
    """
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Находим все PDF-файлы в директории
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    logger.info(f"Всего найдено {len(pdf_files)} PDF-файлов для обработки")
    
    # Проверяем уже обработанные файлы
    processed_files = set()
    if os.path.exists(output_dir):
        processed_files = {f for f in os.listdir(output_dir) if f.lower().endswith('.txt')}
        processed_files = {os.path.splitext(f)[0] for f in processed_files}  # Убираем расширение
    
    logger.info(f"📊 Уже обработано: {len(processed_files)} файлов")
    
    # Фильтруем файлы для обработки
    files_to_process = []
    skipped_files = []
    
    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(pdf_file)[0]
        output_path = os.path.join(output_dir, pdf_name + '.txt')
        
        if pdf_name in processed_files or os.path.exists(output_path):
            skipped_files.append(pdf_file)
        else:
            files_to_process.append(pdf_file)
    
    logger.info(f"📊 Пропущено (уже обработано): {len(skipped_files)} файлов")
    logger.info(f"📊 Будет обработано: {len(files_to_process)} файлов")
    
    if not files_to_process:
        logger.info("✅ Все файлы уже обработаны!")
        return
    
    # Обрабатываем только новые файлы
    processed_count = 0
    for pdf_file in files_to_process:
        input_path = os.path.join(input_dir, pdf_file)
        output_path = os.path.join(output_dir, os.path.splitext(pdf_file)[0] + '.txt')
        
        logger.info(f"📄 Обработка: {pdf_file}")
        convert_pdf_to_text(input_path, output_path)
        
        if os.path.exists(output_path):
            processed_count += 1
    
    logger.info(f"✅ Обработка завершена!")
    logger.info(f"📊 Статистика:")
    logger.info(f"   ✅ Успешно обработано: {processed_count} новых файлов")
    logger.info(f"   ⏭️  Пропущено (уже обработано): {len(skipped_files)} файлов")
    logger.info(f"   📁 Всего найдено: {len(pdf_files)} файлов")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Конвертер PDF в текст')
    parser.add_argument('--input-dir', help='Директория с исходными PDF-файлами')
    parser.add_argument('--output-dir', help='Директория для сохранения извлеченных текстов')
    parser.add_argument('--input-file', help='Отдельный PDF файл для обработки')
    
    args = parser.parse_args()
    
    if args.input_file:
        # Обработка одного файла
        if not args.output_dir:
            parser.error("--output-dir требуется при использовании --input-file")
        
        output_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input_file))[0] + '.txt')
        convert_pdf_to_text(args.input_file, output_path)
    elif args.input_dir and args.output_dir:
        # Обработка директории
        process_all_pdfs(args.input_dir, args.output_dir) 
    else:
        parser.error("Необходимо указать либо --input-file и --output-dir, либо --input-dir и --output-dir") 