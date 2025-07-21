#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Демонстрационный скрипт для показа работы системы генерации мотивировочной части
решения суда с использованием модели QVikhr
"""

import os
import sys
from pathlib import Path

# Добавляем src в путь для импорта модулей
sys.path.append(str(Path(__file__).parent / "src"))

from inference import load_model, generate
from loguru import logger

def demo_qvikhr():
    """
    Демонстрация работы модели QVikhr на примере юридического дела
    """
    print("="*80)
    print("ДЕМОНСТРАЦИЯ СИСТЕМЫ ГЕНЕРАЦИИ МОТИВИРОВОЧНОЙ ЧАСТИ")
    print("Модель: QVikhr (дообученная на юридических данных)")
    print("="*80)
    
    # Проверка существования модели
    model_path = "models/legal_model"
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена по пути: {model_path}")
        print("Убедитесь, что модель QVikhr загружена в директорию models/legal_model")
        return
    
    try:
        # Загрузка модели
        print("🔄 Загрузка модели QVikhr...")
        model, tokenizer = load_model(model_path)
        print("✅ Модель успешно загружена!")
        
        # Пример юридического дела
        facts = """
        Истец Иванов И.И. обратился в суд с иском к ответчику Петрову П.П. о взыскании задолженности по договору займа в размере 150 000 рублей, а также процентов за пользование чужими денежными средствами в размере 15 000 рублей.
        
        В обоснование иска истец указал, что 15 января 2023 года между сторонами был заключен договор займа, по условиям которого истец передал ответчику денежные средства в размере 150 000 рублей сроком на 3 месяца. Срок возврата займа истек 15 апреля 2023 года, однако ответчик не возвратил заемные средства в установленный срок.
        
        Ответчик в судебное заседание не явился, о времени и месте судебного заседания был извещен надлежащим образом. В письменных возражениях ответчик не отрицал факт получения займа, но просил отсрочить его возврат в связи с временными финансовыми трудностями.
        """
        
        print("\n📋 ФАКТИЧЕСКИЕ ОБСТОЯТЕЛЬСТВА ДЕЛА:")
        print("-" * 80)
        print(facts.strip())
        print("-" * 80)
        
        # Генерация мотивировочной части
        print("\n🤖 ГЕНЕРАЦИЯ МОТИВИРОВОЧНОЙ ЧАСТИ...")
        print("⏳ Пожалуйста, подождите...")
        
        reasoning = generate(
            model=model,
            tokenizer=tokenizer,
            facts=facts,
            max_input_length=1024,
            max_output_length=1024,
            num_beams=4,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        print("\n✅ МОТИВИРОВОЧНАЯ ЧАСТЬ СГЕНЕРИРОВАНА:")
        print("=" * 80)
        print(reasoning)
        print("=" * 80)
        
        # Сохранение результата
        output_file = "demo_result.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ФАКТИЧЕСКИЕ ОБСТОЯТЕЛЬСТВА ДЕЛА:\n")
            f.write("-" * 80 + "\n")
            f.write(facts.strip())
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("МОТИВИРОВОЧНАЯ ЧАСТЬ:\n")
            f.write("-" * 80 + "\n")
            f.write(reasoning)
        
        print(f"\n💾 Результат сохранен в файл: {output_file}")
        
    except Exception as e:
        print(f"❌ Ошибка при работе с моделью: {str(e)}")
        logger.error(f"Ошибка демонстрации: {str(e)}")

def main():
    """
    Главная функция демонстрации
    """
    print("🚀 Запуск демонстрации системы LLM-Lawyer")
    print("Модель: QVikhr (4.02B параметров)")
    print("Специализация: Генерация мотивировочной части судебных решений")
    print()
    
    demo_qvikhr()
    
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("="*80)
    print("\nДля использования системы:")
    print("1. Обучение: python src/train.py --train_file data/train_dataset.jsonl --test_file data/train_dataset_test.jsonl")
    print("2. Генерация: python src/inference.py --input_file input.txt --output_file output.txt")
    print("3. Тестирование: python src/test_example.py --test_file data/train_dataset_test.jsonl")
    print("4. GUI: python gui/app.py")

if __name__ == '__main__':
    main() 