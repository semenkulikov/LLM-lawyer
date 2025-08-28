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

class LegalAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Юридический ассистент - Генератор мотивировки")
        self.root.geometry("1000x700")
        
        # Переменные
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        self.setup_ui()
        self.load_model_async()
    
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
        main_frame.rowconfigure(4, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Генератор мотивировочной части судебного решения", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Статус модели
        self.status_label = ttk.Label(main_frame, text="Загрузка модели...", foreground="orange")
        self.status_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Фрейм для ввода
        input_frame = ttk.LabelFrame(main_frame, text="Фактические обстоятельства дела", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        
        # Текстовое поле для ввода
        self.input_text = scrolledtext.ScrolledText(input_frame, height=10, wrap=tk.WORD)
        self.input_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Кнопки для ввода
        input_buttons_frame = ttk.Frame(input_frame)
        input_buttons_frame.grid(row=1, column=0, pady=(10, 0))
        
        ttk.Button(input_buttons_frame, text="Загрузить из файла", 
                  command=self.load_input_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(input_buttons_frame, text="Очистить", 
                  command=self.clear_input).pack(side=tk.LEFT)
        
        # Кнопка генерации
        self.generate_button = ttk.Button(main_frame, text="Сгенерировать мотивировку", 
                                        command=self.generate_reasoning, state="disabled")
        self.generate_button.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Фрейм для вывода
        output_frame = ttk.LabelFrame(main_frame, text="Сгенерированная мотивировка", padding="10")
        output_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Текстовое поле для вывода
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Кнопки для вывода
        output_buttons_frame = ttk.Frame(output_frame)
        output_buttons_frame.grid(row=1, column=0, pady=(10, 0))
        
        ttk.Button(output_buttons_frame, text="Сохранить в файл", 
                  command=self.save_output_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(output_buttons_frame, text="Очистить", 
                  command=self.clear_output).pack(side=tk.LEFT)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
    
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
    
    def clear_output(self):
        """Очистка выходного текста"""
        self.output_text.delete(1.0, tk.END)
    
    def generate_reasoning(self):
        """Генерация мотивировки"""
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
        
        def generate_thread():
            try:
                reasoning = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    facts=facts,
                    max_input_length=1024,
                    max_output_length=1024
                )
                self.root.after(0, self.on_generation_complete, reasoning)
            except Exception as e:
                self.root.after(0, self.on_generation_error, str(e))
        
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()
    
    def on_generation_complete(self, reasoning):
        """Обработчик завершения генерации"""
        self.progress.stop()
        self.generate_button.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(1.0, reasoning)
    
    def on_generation_error(self, error_msg):
        """Обработчик ошибки генерации"""
        self.progress.stop()
        self.generate_button.config(state="normal")
        messagebox.showerror("Ошибка генерации", f"Не удалось сгенерировать мотивировку: {error_msg}")
    
    def save_output_file(self):
        """Сохранение результата в файл"""
        filename = filedialog.asksaveasfilename(
            title="Сохранить мотивировку",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if filename:
            try:
                content = self.output_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Успех", "Мотивировка сохранена в файл")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

def main():
    root = tk.Tk()
    app = LegalAssistantGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main() 