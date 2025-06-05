import os
import json
import logging
from collections import Counter
from typing import Dict, List, Tuple

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def analyze_structured_files(directory: str) -> Dict:
    """
    Анализирует все JSON-файлы с разделенными текстами в указанной директории.
    
    Args:
        directory: Путь к директории с JSON-файлами
        
    Returns:
        Словарь со статистикой и результатами анализа
    """
    results = {
        "total_files": 0,
        "quality_stats": Counter(),
        "section_stats": {
            "facts": {"count": 0, "avg_length": 0, "empty": 0},
            "reasoning": {"count": 0, "avg_length": 0, "empty": 0},
            "conclusion": {"count": 0, "avg_length": 0, "empty": 0}
        },
        "problems": []
    }
    
    # Получаем список всех JSON-файлов
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    results["total_files"] = len(json_files)
    
    # Анализируем каждый файл
    for json_file in json_files:
        try:
            file_path = os.path.join(directory, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Собираем статистику по качеству
            quality = data.get("quality", "неизвестно")
            results["quality_stats"][quality] += 1
            
            # Анализируем каждую секцию
            sections = data.get("sections", {})
            for section_name in ["facts", "reasoning", "conclusion"]:
                section_content = sections.get(section_name, "")
                section_length = len(section_content)
                
                if section_length > 0:
                    results["section_stats"][section_name]["count"] += 1
                    results["section_stats"][section_name]["avg_length"] += section_length
                else:
                    results["section_stats"][section_name]["empty"] += 1
                    # Записываем проблему, если секция пуста
                    results["problems"].append({
                        "file": json_file,
                        "problem": f"Пустая секция '{section_name}'"
                    })
            
            # Проверяем дополнительные проблемы
            if sections.get("facts", "") and sections.get("reasoning", "") and len(sections.get("facts", "")) > len(sections.get("reasoning", "")):
                results["problems"].append({
                    "file": json_file,
                    "problem": "Секция facts длиннее, чем reasoning"
                })
                
        except Exception as e:
            results["problems"].append({
                "file": json_file,
                "problem": f"Ошибка при анализе файла: {str(e)}"
            })
    
    # Вычисляем средние длины для секций
    for section_name in ["facts", "reasoning", "conclusion"]:
        section_count = results["section_stats"][section_name]["count"]
        if section_count > 0:
            total_length = results["section_stats"][section_name]["avg_length"]
            results["section_stats"][section_name]["avg_length"] = total_length / section_count
    
    return results

def print_report(results: Dict) -> None:
    """
    Выводит отчет о качестве разделения текстов на основе результатов анализа.
    
    Args:
        results: Словарь с результатами анализа
    """
    print("\n" + "="*50)
    print(f"ОТЧЕТ О КАЧЕСТВЕ РАЗДЕЛЕНИЯ ТЕКСТОВ")
    print("="*50)
    
    # Общая статистика
    print(f"\nВсего проанализировано файлов: {results['total_files']}")
    
    # Статистика по качеству
    print("\nСтатистика по качеству разделения:")
    total = sum(results["quality_stats"].values())
    for quality, count in sorted(results["quality_stats"].items()):
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {quality}: {count} ({percentage:.1f}%)")
    
    # Статистика по секциям
    print("\nСтатистика по секциям:")
    for section_name, stats in results["section_stats"].items():
        empty_percent = (stats["empty"] / results["total_files"]) * 100 if results["total_files"] > 0 else 0
        print(f"  {section_name}:")
        print(f"    Найдено: {stats['count']} ({(stats['count']/results['total_files'])*100:.1f}%)")
        print(f"    Пусто: {stats['empty']} ({empty_percent:.1f}%)")
        print(f"    Средняя длина: {stats['avg_length']:.0f} символов")
    
    # Топ проблем
    if results["problems"]:
        print("\nОбнаруженные проблемы:")
        problem_counts = Counter([p["problem"] for p in results["problems"]])
        for problem, count in problem_counts.most_common(10):
            print(f"  {problem}: {count} файлов")
        
        # Детальный список проблемных файлов (опционально)
        print("\nПримеры проблемных файлов:")
        problem_types = set([p["problem"] for p in results["problems"]])
        for problem_type in list(problem_types)[:5]:  # Ограничиваем 5 типами проблем
            files_with_problem = [p["file"] for p in results["problems"] if p["problem"] == problem_type]
            print(f"  {problem_type}:")
            for file in files_with_problem[:3]:  # Показываем до 3 примеров
                print(f"    - {file}")
    else:
        print("\nПроблем не обнаружено.")
    
    print("\n" + "="*50)

def main():
    """
    Основная функция для запуска анализа.
    """
    try:
        # Путь к директории с JSON-файлами
        structured_dir = os.path.join("data", "structured")
        
        if not os.path.exists(structured_dir):
            logging.error(f"Директория {structured_dir} не существует")
            return
        
        # Анализируем файлы
        logging.info(f"Начинаем анализ файлов в директории {structured_dir}")
        results = analyze_structured_files(structured_dir)
        
        # Выводим отчет
        print_report(results)
        
        # Сохраняем отчет в JSON (опционально)
        output_file = os.path.join("data", "quality_report.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Отчет сохранен в {output_file}")
        
    except Exception as e:
        logging.error(f"Ошибка при выполнении анализа: {str(e)}")

if __name__ == "__main__":
    main() 