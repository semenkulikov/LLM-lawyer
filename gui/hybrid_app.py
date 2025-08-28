#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import sys
import os
import threading
import json
from loguru import logger

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import load_model, generate
from hybrid_processor import create_hybrid_processor

class HybridLegalAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Юридический ассистент - Универсальная гибридная система")
        self.root.geometry("1400x900")
        
        # Переменные
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.hybrid_processor = None
        self.hybrid_enabled = True
        self.selected_provider = "gemini"  # По умолчанию Gemini
        
        self.setup_ui()
        self.load_model_async()
        self.init_hybrid_processor()
    
    def setup_ui(self):
        """Настройка интерфейса"""
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Конфигурация сетки
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Универсальная гибридная система", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Подзаголовок
        subtitle_label = ttk.Label(main_frame, text="Локальная модель QVikhr + внешний LLM (Gemini/OpenAI)", 
                                   font=("Arial", 12), foreground="blue")
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Статус модели
        self.status_label = ttk.Label(main_frame, text="Загрузка модели...", foreground="orange")
        self.status_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        
        # Фрейм настроек гибридного режима
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки обработки", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Включение/выключение гибридного режима
        self.hybrid_var = tk.BooleanVar(value=True)
        hybrid_check = ttk.Checkbutton(settings_frame, text="Включить гибридный режим", 
                                      variable=self.hybrid_var, command=self.toggle_hybrid_mode)
        hybrid_check.pack(side=tk.LEFT, padx=(0, 20))
        
        # Выбор провайдера
        ttk.Label(settings_frame, text="Провайдер:").pack(side=tk.LEFT, padx=(0, 5))
        self.provider_var = tk.StringVar(value="gemini")
        provider_combo = ttk.Combobox(settings_frame, textvariable=self.provider_var, 
                                      values=["gemini", "openai"], 
                                      state="readonly", width=10)
        provider_combo.pack(side=tk.LEFT, padx=(0, 10))
        provider_combo.bind('<<ComboboxSelected>>', self.on_provider_change)
        
        # Выбор режима обработки
        ttk.Label(settings_frame, text="Режим:").pack(side=tk.LEFT, padx=(0, 5))
        self.mode_var = tk.StringVar(value="polish")
        mode_combo = ttk.Combobox(settings_frame, textvariable=self.mode_var, 
                                  values=["polish", "enhance", "verify"], 
                                  state="readonly", width=15)
        mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Описание режимов
        mode_descriptions = {
            "polish": "Полировка текста",
            "enhance": "Расширение документа", 
            "verify": "Проверка и исправление"
        }
        self.mode_desc_label = ttk.Label(settings_frame, text=mode_descriptions["polish"], 
                                        foreground="gray")
        self.mode_desc_label.pack(side=tk.LEFT)
        
        # Привязка события изменения режима
        mode_combo.bind('<<ComboboxSelected>>', self.on_mode_change)
        
        # Фрейм для ввода
        input_frame = ttk.LabelFrame(main_frame, text="Фактические обстоятельства дела", padding="10")
        input_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        
        # Текстовое поле для ввода
        self.input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD)
        self.input_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Кнопки для ввода
        input_buttons_frame = ttk.Frame(input_frame)
        input_buttons_frame.grid(row=1, column=0, pady=(10, 0))
        
        ttk.Button(input_buttons_frame, text="Загрузить из файла", 
                   command=self.load_input_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(input_buttons_frame, text="Очистить", 
                   command=self.clear_input).pack(side=tk.LEFT)
        
        # Кнопка генерации
        self.generate_button = ttk.Button(main_frame, text="Сгенерировать документ с гибридной обработкой", 
                                         command=self.generate_reasoning, state="disabled")
        self.generate_button.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Фрейм для вывода с вкладками
        output_notebook = ttk.Notebook(main_frame)
        output_notebook.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Вкладка локального результата
        local_frame = ttk.Frame(output_notebook)
        output_notebook.add(local_frame, text="Локальная модель (QVikhr)")
        local_frame.columnconfigure(0, weight=1)
        local_frame.rowconfigure(0, weight=1)
        
        self.local_output_text = scrolledtext.ScrolledText(local_frame, height=12, wrap=tk.WORD)
        self.local_output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Вкладка гибридного результата
        hybrid_frame = ttk.Frame(output_notebook)
        output_notebook.add(hybrid_frame, text="Гибридный результат")
        hybrid_frame.columnconfigure(0, weight=1)
        hybrid_frame.rowconfigure(0, weight=1)
        
        self.hybrid_output_text = scrolledtext.ScrolledText(hybrid_frame, height=12, wrap=tk.WORD)
        self.hybrid_output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Кнопки для вывода
        output_buttons_frame = ttk.Frame(main_frame)
        output_buttons_frame.grid(row=7, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(output_buttons_frame, text="Сохранить локальный результат", 
                   command=lambda: self.save_output_file("local")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(output_buttons_frame, text="Сохранить гибридный результат", 
                   command=lambda: self.save_output_file("hybrid")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(output_buttons_frame, text="Очистить все", 
                   command=self.clear_all_outputs).pack(side=tk.LEFT)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Статус обработки
        self.processing_status = ttk.Label(main_frame, text="", foreground="blue")
        self.processing_status.grid(row=9, column=0, columnspan=3, pady=(5, 0))
    
    def init_hybrid_processor(self):
        """Инициализация гибридного процессора"""
        try:
            self.hybrid_processor = create_hybrid_processor(provider=self.selected_provider)
            logger.info(f"Гибридный процессор инициализирован с провайдером: {self.selected_provider}")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать гибридный процессор: {e}")
            self.hybrid_processor = None
            self.hybrid_var.set(False)
    
    def on_provider_change(self, event=None):
        """Обработчик изменения провайдера"""
        self.selected_provider = self.provider_var.get()
        logger.info(f"Выбран провайдер: {self.selected_provider}")
        
        # Переинициализируем процессор
        try:
            self.hybrid_processor = create_hybrid_processor(provider=self.selected_provider)
            logger.info(f"Гибридный процессор переинициализирован с провайдером: {self.selected_provider}")
        except Exception as e:
            logger.error(f"Ошибка переинициализации процессора: {e}")
            messagebox.showerror("Ошибка", f"Не удалось инициализировать {self.selected_provider}: {e}")
    
    def toggle_hybrid_mode(self):
        """Переключение гибридного режима"""
        self.hybrid_enabled = self.hybrid_var.get()
        if self.hybrid_enabled and not self.hybrid_processor:
            messagebox.showwarning("Предупреждение", 
                                  f"Гибридный процессор недоступен. Проверьте настройки {self.selected_provider} API.")
            self.hybrid_var.set(False)
            self.hybrid_enabled = False
    
    def on_mode_change(self, event=None):
        """Обработчик изменения режима обработки"""
        mode_descriptions = {
            "polish": "Полировка текста - исправление ошибок и улучшение стиля",
            "enhance": "Расширение документа - добавление недостающих элементов", 
            "verify": "Проверка и исправление - проверка на ошибки с комментариями"
        }
        current_mode = self.mode_var.get()
        self.mode_desc_label.config(text=mode_descriptions.get(current_mode, ""))
    
    def load_model_async(self):
        """Асинхронная загрузка модели"""
        def load():
            try:
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'legal_model')
                if os.path.exists(model_path):
                    self.model, self.tokenizer = load_model(model_path)
                    self.model_loaded = True
                    self.root.after(0, self.on_model_loaded)
                else:
                    self.root.after(0, self.on_model_error, "Модель не найдена. Обучите модель сначала.")
            except Exception as e:
                self.root.after(0, self.on_model_error, str(e))
        
        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self):
        """Обработчик успешной загрузки модели"""
        self.status_label.config(text="Модель загружена ✓", foreground="green")
        self.generate_button.config(state="normal")
    
    def on_model_error(self, error_msg):
        """Обработчик ошибки загрузки модели"""
        self.status_label.config(text=f"Ошибка загрузки модели: {error_msg}", foreground="red")
    
    def load_input_file(self):
        """Загрузка входного файла"""
        filename = filedialog.askopenfilename(
            title="Выберите файл с фактическими обстоятельствами",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.input_text.delete(1.0, tk.END)
                self.input_text.insert(1.0, content)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")
    
    def clear_input(self):
        """Очистка входного текста"""
        self.input_text.delete(1.0, tk.END)
    
    def clear_all_outputs(self):
        """Очистка всех выходных текстовых полей"""
        self.local_output_text.delete(1.0, tk.END)
        self.hybrid_output_text.delete(1.0, tk.END)
    
    def generate_reasoning(self):
        """Генерация документа с гибридной обработкой"""
        if not self.model_loaded:
            messagebox.showerror("Ошибка", "Модель не загружена")
            return
        
        facts = self.input_text.get(1.0, tk.END).strip()
        if not facts:
            messagebox.showwarning("Предупреждение", "Введите фактические обстоятельства дела")
            return
        
        # Запуск генерации в отдельном потоке
        self.generate_button.config(state="disabled")
        self.progress.start()
        self.processing_status.config(text="Генерация локальной моделью QVikhr...")
        
        def generate_thread():
            try:
                # Шаг 1: Генерация локальной моделью
                self.root.after(0, lambda: self.processing_status.config(text="Генерация локальной моделью QVikhr..."))
                local_response = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    facts=facts,
                    max_input_length=1024,
                    max_output_length=1024
                )
                
                # Отображаем локальный результат
                self.root.after(0, lambda: self.local_output_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.local_output_text.insert(1.0, local_response))
                
                # Шаг 2: Гибридная обработка (если включена)
                if self.hybrid_enabled and self.hybrid_processor:
                    provider_name = self.selected_provider.upper()
                    self.root.after(0, lambda: self.processing_status.config(text=f"Обработка через {provider_name}..."))
                    
                    mode = self.mode_var.get()
                    hybrid_response = self.hybrid_processor.process_with_external_llm(
                        local_response=local_response,
                        original_query=facts,
                        mode=mode
                    )
                    
                    # Отображаем результат гибридной обработки
                    self.root.after(0, lambda: self.hybrid_output_text.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.hybrid_output_text.insert(1.0, hybrid_response))
                    
                    self.root.after(0, lambda: self.processing_status.config(text=f"Готово! {provider_name} обработка завершена."))
                else:
                    # Если гибридный режим отключен, копируем локальный результат
                    self.root.after(0, lambda: self.hybrid_output_text.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.hybrid_output_text.insert(1.0, local_response))
                    self.root.after(0, lambda: self.processing_status.config(text="Готово! Только локальная модель QVikhr."))
                
                # Завершение
                self.root.after(0, self.on_generation_complete)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_generation_error(str(e)))
        
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()
    
    def on_generation_complete(self):
        """Обработчик завершения генерации"""
        self.progress.stop()
        self.generate_button.config(state="normal")
        # Статус уже обновлен в потоке генерации
    
    def on_generation_error(self, error_msg):
        """Обработчик ошибки генерации"""
        self.progress.stop()
        self.generate_button.config(state="normal")
        self.processing_status.config(text="Ошибка генерации!")
        messagebox.showerror("Ошибка генерации", f"Не удалось сгенерировать документ: {error_msg}")
    
    def save_output_file(self, source_type):
        """Сохранение результата в файл"""
        filename = filedialog.asksaveasfilename(
            title="Сохранить документ",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if filename:
            try:
                if source_type == "local":
                    content = self.local_output_text.get(1.0, tk.END)
                elif source_type == "hybrid":
                    content = self.hybrid_output_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Успех", f"Документ сохранен из {source_type} модели")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

def main():
    root = tk.Tk()
    app = HybridLegalAssistantGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
