#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

class DatasetCleaner:
    """Очистка и подготовка датасета для обучения с LoRA"""
    
    def __init__(self):
        # Разнообразные инструкции для обучения
        self.instructions = [
            "Составь исковое заявление по фактическим обстоятельствам дела.",
            "Сформулируй юридическое заявление на основе описания ситуации.",
            "Напиши исковое заявление, исходя из приведённых фактов.",
            "Составь иск по указанным обстоятельствам.",
            "Создай исковое заявление на основании описанных событий.",
            "Подготовь юридический документ по представленным фактам.",
            "Составь заявление в суд на основе описанной ситуации."
        ]
        
        # Ключевые фразы для валидации иска
        self.valid_phrases = [
            "ИСКОВОЕ ЗАЯВЛЕНИЕ",
            "МОТИВИРОВОЧНАЯ ЧАСТЬ", 
            "РЕЗОЛЮТИВНАЯ ЧАСТЬ",
            "В СУД",
            "ИСТЕЦ:",
            "ОТВЕТЧИК:",
            "ПРОШУ:",
            "НА ОСНОВАНИИ ИЗЛОЖЕННОГО"
        ]
        
        # Фразы для фильтрации мусора
        self.invalid_phrases = [
            "Уважаемый Владимир Владимирович",
            "Дорогой пользователь",
            "Спасибо за обращение",
            "Надеюсь, это поможет",
            "Если у вас есть вопросы",
            "Обратитесь к юристу",
            "Это не юридическая консультация"
        ]
    
    def clean_dataset(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Очистка датасета
        
        Args:
            input_file: Входной файл датасета
            output_file: Выходной очищенный файл
            
        Returns:
            Статистика очистки
        """
        logger.info(f"Начинаю очистку датасета: {input_file}")
        
        cleaned_examples = []
        total_examples = 0
        filtered_examples = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                total_examples += 1
                
                try:
                    example = json.loads(line)
                    
                    # Проверяем структуру
                    if not self._validate_structure(example):
                        logger.warning(f"Пропущена запись {line_num}: неверная структура")
                        filtered_examples += 1
                        continue
                    
                    # Очищаем пример
                    cleaned_example = self._clean_example(example)
                    
                    if cleaned_example:
                        cleaned_examples.append(cleaned_example)
                    else:
                        filtered_examples += 1
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Ошибка JSON в строке {line_num}: {e}")
                    filtered_examples += 1
                    continue
        
        # Сохраняем очищенный датасет
        self._save_cleaned_dataset(cleaned_examples, output_file)
        
        # Статистика
        stats = {
            'total_examples': total_examples,
            'cleaned_examples': len(cleaned_examples),
            'filtered_examples': filtered_examples,
            'cleaning_rate': len(cleaned_examples) / total_examples if total_examples > 0 else 0
        }
        
        logger.info(f"""
📊 Статистика очистки датасета:

📝 Всего примеров: {stats['total_examples']}
✅ Очищено и сохранено: {stats['cleaned_examples']}
❌ Отфильтровано: {stats['filtered_examples']}
📈 Процент очистки: {stats['cleaning_rate']:.1%}

📁 Очищенный датасет: {output_file}
        """)
        
        return stats
    
    def _validate_structure(self, example: Dict[str, Any]) -> bool:
        """Проверка структуры примера"""
        required_fields = ['instruction', 'input', 'output']
        return all(field in example for field in required_fields)
    
    def _clean_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Очистка отдельного примера"""
        output = example.get('output', '').strip()
        
        # Проверяем на валидность иска
        if not self._is_valid_legal_document(output):
            return None
        
        # Проверяем на мусор
        if self._contains_garbage(output):
            return None
        
        # Очищаем текст
        cleaned_output = self._clean_text(output)
        
        if not cleaned_output or len(cleaned_output) < 100:
            return None
        
        # Создаем новый пример с улучшенной инструкцией
        cleaned_example = {
            'instruction': self._get_random_instruction(),
            'input': example.get('input', '').strip(),
            'output': cleaned_output
        }
        
        # Добавляем метаданные если есть
        if 'metadata' in example:
            cleaned_example['metadata'] = example['metadata']
        
        return cleaned_example
    
    def _is_valid_legal_document(self, text: str) -> bool:
        """Проверка на валидность юридического документа"""
        text_upper = text.upper()
        
        # Должна содержать хотя бы 2 ключевые фразы
        valid_count = sum(1 for phrase in self.valid_phrases if phrase in text_upper)
        
        # Минимальная длина
        if len(text) < 200:
            return False
        
        # Должна содержать мотивировочную часть
        if "МОТИВИРОВОЧНАЯ ЧАСТЬ" not in text_upper:
            return False
        
        return valid_count >= 2
    
    def _contains_garbage(self, text: str) -> bool:
        """Проверка на наличие мусора"""
        text_lower = text.lower()
        
        for phrase in self.invalid_phrases:
            if phrase.lower() in text_lower:
                return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста от лишних элементов"""
        # Убираем лишние пробелы и переносы
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Убираем лишние символы в начале и конце
        text = text.strip()
        
        # Убираем комментарии в квадратных скобках
        text = re.sub(r'\[.*?\]', '', text)
        
        # Убираем лишние кавычки
        text = re.sub(r'[""]', '"', text)
        
        return text.strip()
    
    def _get_random_instruction(self) -> str:
        """Получение случайной инструкции"""
        import random
        return random.choice(self.instructions)
    
    def _save_cleaned_dataset(self, examples: List[Dict[str, Any]], output_file: str):
        """Сохранение очищенного датасета"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Очищенный датасет сохранен: {output_file}")

def create_lora_training_config(dataset_path: str, output_dir: str = "models/legal_model_lora") -> Dict[str, Any]:
    """Создание конфигурации для обучения с LoRA"""
    
    config = {
        "dataset_path": dataset_path,
        "model_name": "Vikhrmodels/QVikhr-3-4B-Instruction",  # Используем вашу модель
        "training_args": {
            "output_dir": output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,  # Маленький батч для 8GB GPU
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,  # Рекомендуется для LoRA
            "warmup_steps": 50,
            "logging_steps": 10,
            "save_steps": 200,
            "eval_steps": 200,
            "save_total_limit": 2,
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "data_config": {
            "max_length": 2048,
            "truncation": True,
            "padding": "max_length"
        }
    }
    
    return config

def main():
    """Основная функция очистки датасета"""
    input_file = "datasets/merged_training_dataset.jsonl"
    output_file = "datasets/clean_training_dataset.jsonl"
    
    if not Path(input_file).exists():
        logger.error(f"Файл {input_file} не найден!")
        return
    
    # Очищаем датасет
    cleaner = DatasetCleaner()
    stats = cleaner.clean_dataset(input_file, output_file)
    
    if stats['cleaned_examples'] > 0:
        # Создаем конфигурацию для LoRA
        config = create_lora_training_config(output_file)
        
        config_file = "datasets/lora_training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"""
🎉 Очистка завершена! 

📊 Результат:
- Очищено примеров: {stats['cleaned_examples']}
- Процент очистки: {stats['cleaning_rate']:.1%}

🚀 Для обучения с LoRA используйте:
python src/train_lora.py --config {config_file}

💡 Рекомендации:
- Если примеров < 1000, соберите больше данных
- Для лучшего качества используйте 3-5 эпох
- Мониторьте eval_loss для предотвращения переобучения
        """)
    else:
        logger.error("❌ Не удалось очистить ни одного примера!")

if __name__ == "__main__":
    main()
