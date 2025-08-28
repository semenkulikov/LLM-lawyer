import os
from pathlib import Path

# Проверяем файлы
text_files = list(Path('data/processed').glob('*.txt'))
output_dir = 'data/analyzed'

print(f'Всего txt файлов: {len(text_files)}')
print(f'Первые 5: {[f.stem for f in text_files[:5]]}')

# Проверяем уже обработанные
processed_files = set()
if os.path.exists(output_dir):
    processed_files = {f for f in os.listdir(output_dir) if f.lower().endswith('_analyzed.json')}
    processed_files = {f.replace('_analyzed.json', '') for f in processed_files}

print(f'Уже обработано: {len(processed_files)}')
print(f'Примеры обработанных: {list(processed_files)[:5]}')

# Проверяем логику фильтрации
files_to_process = []
skipped_files = []

for text_file in text_files[:10]:
    file_name = text_file.stem
    output_file = os.path.join(output_dir, f'{file_name}_analyzed.json')
    
    if file_name in processed_files or os.path.exists(output_file):
        print(f'Пропущен: {file_name}')
        skipped_files.append(str(text_file))
    else:
        print(f'Будет обработан: {file_name}')
        files_to_process.append(str(text_file))

print(f'Итого будет обработано: {len(files_to_process)}')
print(f'Пропущено: {len(skipped_files)}') 