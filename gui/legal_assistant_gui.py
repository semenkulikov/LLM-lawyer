#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import threading
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                             QTextEdit, QComboBox, QCheckBox, QProgressBar,
                             QFrame, QFileDialog, QMessageBox, QTabWidget,
                             QSplitter, QGroupBox, QScrollArea, QTextBrowser)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon, QLinearGradient, QPainter
from loguru import logger

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import load_model, generate
from hybrid_processor import create_hybrid_processor

# Для поддержки Markdown
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    logger.warning("Markdown не установлен. Установите: pip install markdown")

class ModelLoaderThread(QThread):
    """Поток для загрузки модели"""
    model_loaded = pyqtSignal(object, object)
    model_error = pyqtSignal(str)
    
    def run(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'legal_model')
            if os.path.exists(model_path):
                model, tokenizer = load_model(model_path)
                self.model_loaded.emit(model, tokenizer)
            else:
                self.model_error.emit("Модель не найдена. Обучите модель сначала.")
        except Exception as e:
            self.model_error.emit(str(e))

class GenerationThread(QThread):
    """Поток для генерации текста"""
    generation_complete = pyqtSignal(str, str)  # local_response, hybrid_response
    generation_error = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)  # progress, status
    
    def __init__(self, model, tokenizer, facts, hybrid_processor, mode):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.facts = facts
        self.hybrid_processor = hybrid_processor
        self.mode = mode
    
    def run(self):
        try:
            # Генерация локальной моделью
            self.progress_update.emit(25, "Генерация локальной моделью...")
            local_response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                facts=self.facts,
                max_input_length=1024,
                max_output_length=1024
            )
            
            # Гибридная обработка
            if self.hybrid_processor:
                self.progress_update.emit(75, "Обработка внешним LLM...")
                hybrid_response = self.hybrid_processor.process_with_external_llm(
                    local_response=local_response,
                    original_query=self.facts,
                    mode=self.mode
                )
            else:
                hybrid_response = local_response
            
            self.progress_update.emit(100, "Завершено!")
            self.generation_complete.emit(local_response, hybrid_response)
            
        except Exception as e:
            self.generation_error.emit(str(e))

class LegalAssistantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.hybrid_processor = None
        self.hybrid_enabled = True
        self.selected_provider = "openai"
        self.selected_mode = "polish"
        
        self.setup_ui()
        self.setup_styles()
        self.load_model_async()
        self.init_hybrid_processor()
    
    def setup_ui(self):
        """Настройка интерфейса"""
        self.setWindowTitle("Юридический ассистент - Универсальная гибридная система")
        self.setGeometry(100, 100, 1800, 1100)
        self.setMinimumSize(1400, 900)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Заголовок
        self.create_header(main_layout)
        
        # Статус модели
        self.create_status_section(main_layout)
        
        # Настройки
        self.create_settings_section(main_layout)
        
        # Основной контент
        self.create_main_content(main_layout)
        
        # Кнопки управления
        self.create_control_buttons(main_layout)
        
        # Прогресс бар
        self.create_progress_section(main_layout)
    
    def create_header(self, parent_layout):
        """Создание заголовка"""
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setMinimumHeight(120)
        
        header_layout = QVBoxLayout(header_frame)
        
        title_label = QLabel("Юридический ассистент")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        
        subtitle_label = QLabel("Универсальная гибридная система")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        
        parent_layout.addWidget(header_frame)
    
    def create_status_section(self, parent_layout):
        """Создание секции статуса"""
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")
        
        status_layout = QHBoxLayout(status_frame)
        
        # Текст статуса
        self.status_label = QLabel("Загрузка модели...")
        self.status_label.setObjectName("statusLabel")
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        parent_layout.addWidget(status_frame)
    
    def create_settings_section(self, parent_layout):
        """Создание секции настроек"""
        settings_group = QGroupBox("Настройки обработки")
        settings_group.setObjectName("settingsGroup")
        
        settings_layout = QHBoxLayout(settings_group)
        
        # Гибридный режим
        self.hybrid_checkbox = QCheckBox("Включить гибридный режим")
        self.hybrid_checkbox.setChecked(True)
        self.hybrid_checkbox.toggled.connect(self.toggle_hybrid_mode)
        self.hybrid_checkbox.setObjectName("customCheckBox")
        
        # Провайдер
        provider_label = QLabel("Провайдер:")
        provider_label.setObjectName("settingsLabel")
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["OpenAI", "Gemini"])
        self.provider_combo.setCurrentText("OpenAI")
        self.provider_combo.currentTextChanged.connect(self.on_provider_change)
        self.provider_combo.setObjectName("customComboBox")
        
        # Режим обработки
        mode_label = QLabel("Режим:")
        mode_label.setObjectName("settingsLabel")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Полировка", "Расширение", "Проверка"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
        self.mode_combo.setObjectName("customComboBox")
        
        # Описание режима
        self.mode_desc_label = QLabel("Исправление ошибок и улучшение стиля")
        self.mode_desc_label.setObjectName("modeDescLabel")
        
        settings_layout.addWidget(self.hybrid_checkbox)
        settings_layout.addWidget(provider_label)
        settings_layout.addWidget(self.provider_combo)
        settings_layout.addWidget(mode_label)
        settings_layout.addWidget(self.mode_combo)
        settings_layout.addWidget(self.mode_desc_label)
        settings_layout.addStretch()
        
        parent_layout.addWidget(settings_group)
    
    def create_main_content(self, parent_layout):
        """Создание основного контента"""
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель - ввод
        input_group = QGroupBox("Фактические обстоятельства дела")
        input_group.setObjectName("inputGroup")
        
        input_layout = QVBoxLayout(input_group)
        
        # Кнопки для ввода
        input_buttons_layout = QHBoxLayout()
        
        self.load_file_btn = QPushButton("Загрузить из файла")
        self.load_file_btn.clicked.connect(self.load_input_file)
        self.load_file_btn.setObjectName("actionButton")
        
        self.clear_input_btn = QPushButton("Очистить")
        self.clear_input_btn.clicked.connect(self.clear_input)
        self.clear_input_btn.setObjectName("actionButton")
        
        input_buttons_layout.addWidget(self.load_file_btn)
        input_buttons_layout.addWidget(self.clear_input_btn)
        input_buttons_layout.addStretch()
        
        # Текстовое поле для ввода
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Введите фактические обстоятельства дела...")
        self.input_text.setObjectName("inputTextEdit")
        
        input_layout.addLayout(input_buttons_layout)
        input_layout.addWidget(self.input_text)
        
        # Правая панель - вывод с вкладками
        output_tabs = QTabWidget()
        output_tabs.setObjectName("outputTabs")
        
        # Вкладка локального результата
        local_frame = QWidget()
        local_layout = QVBoxLayout(local_frame)
        
        local_header = QLabel("Локальная модель (QVikhr)")
        local_header.setObjectName("tabHeaderLabel")
        
        self.local_output_text = QTextBrowser()
        self.local_output_text.setOpenExternalLinks(True)
        self.local_output_text.setObjectName("outputTextEdit")
        
        local_layout.addWidget(local_header)
        local_layout.addWidget(self.local_output_text)
        
        # Вкладка гибридного результата
        hybrid_frame = QWidget()
        hybrid_layout = QVBoxLayout(hybrid_frame)
        
        hybrid_header = QLabel("Гибридный результат")
        hybrid_header.setObjectName("tabHeaderLabel")
        
        self.hybrid_output_text = QTextBrowser()
        self.hybrid_output_text.setOpenExternalLinks(True)
        self.hybrid_output_text.setObjectName("outputTextEdit")
        
        hybrid_layout.addWidget(hybrid_header)
        hybrid_layout.addWidget(self.hybrid_output_text)
        
        output_tabs.addTab(local_frame, "Локальная модель")
        output_tabs.addTab(hybrid_frame, "Гибридный результат")
        
        splitter.addWidget(input_group)
        splitter.addWidget(output_tabs)
        splitter.setSizes([500, 900])
        
        parent_layout.addWidget(splitter, 1)
    
    def create_control_buttons(self, parent_layout):
        """Создание кнопок управления"""
        buttons_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Сгенерировать документ")
        self.generate_btn.clicked.connect(self.generate_reasoning)
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.setEnabled(False)
        
        self.save_local_btn = QPushButton("Сохранить локальный")
        self.save_local_btn.clicked.connect(lambda: self.save_output_file("local"))
        self.save_local_btn.setObjectName("secondaryButton")
        
        self.save_hybrid_btn = QPushButton("Сохранить гибридный")
        self.save_hybrid_btn.clicked.connect(lambda: self.save_output_file("hybrid"))
        self.save_hybrid_btn.setObjectName("secondaryButton")
        
        self.clear_all_btn = QPushButton("Очистить все")
        self.clear_all_btn.clicked.connect(self.clear_all_outputs)
        self.clear_all_btn.setObjectName("secondaryButton")
        
        buttons_layout.addWidget(self.generate_btn)
        buttons_layout.addWidget(self.save_local_btn)
        buttons_layout.addWidget(self.save_hybrid_btn)
        buttons_layout.addWidget(self.clear_all_btn)
        
        parent_layout.addLayout(buttons_layout)
    
    def create_progress_section(self, parent_layout):
        """Создание секции прогресса"""
        progress_frame = QFrame()
        progress_frame.setObjectName("progressFrame")
        
        progress_layout = QVBoxLayout(progress_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setObjectName("progressBar")
        
        self.processing_status = QLabel("")
        self.processing_status.setObjectName("processingStatusLabel")
        self.processing_status.setAlignment(Qt.AlignCenter)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.processing_status)
        
        parent_layout.addWidget(progress_frame)
    
    def setup_styles(self):
        """Настройка стилей"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #headerFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2c3e50, stop:1 #34495e);
                border-radius: 15px;
                margin: 10px;
            }
            
            #titleLabel {
                color: white;
                font-size: 32px;
                font-weight: bold;
                margin: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #subtitleLabel {
                color: #bdc3c7;
                font-size: 18px;
                margin: 5px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #statusFrame {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
            }
            
            #statusLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #settingsGroup {
                background: white;
                border: 2px solid #3498db;
                border-radius: 10px;
                font-weight: bold;
                padding: 20px;
            }
            
            #settingsGroup::title {
                color: #3498db;
                font-size: 18px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #settingsLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #inputGroup, #outputTabs {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
            }
            
            #inputGroup::title, #outputTabs::title {
                color: #2c3e50;
                font-size: 18px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #inputTextEdit, #outputTextEdit {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                font-size: 14px;
                background: #fafafa;
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.6;
            }
            
            #inputTextEdit:focus, #outputTextEdit:focus {
                border-color: #3498db;
                background: white;
            }
            
            #primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #27ae60, stop:1 #2ecc71);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                min-width: 200px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2ecc71, stop:1 #27ae60);
            }
            
            #primaryButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
            
            #secondaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #secondaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2980b9, stop:1 #3498db);
            }
            
            #actionButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f39c12, stop:1 #e67e22);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #actionButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e67e22, stop:1 #f39c12);
            }
            
            #progressFrame {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
            }
            
            #progressBar {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                text-align: center;
                background: #f0f0f0;
                height: 20px;
            }
            
            #progressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #27ae60, stop:1 #2ecc71);
                border-radius: 8px;
            }
            
            #processingStatusLabel {
                font-size: 16px;
                font-weight: bold;
                color: #3498db;
                margin-top: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #tabHeaderLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #modeDescLabel {
                color: #7f8c8d;
                font-style: italic;
                margin-left: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #customComboBox {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                padding: 10px;
                background: white;
                font-size: 14px;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-width: 120px;
            }
            
            #customComboBox:focus {
                border-color: #3498db;
            }
            
            #customComboBox::drop-down {
                border: none;
                width: 25px;
            }
            
            #customComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid #7f8c8d;
            }
            
            #customCheckBox {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #customCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background: white;
            }
            
            #customCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-color: #3498db;
            }
            
            QTabWidget::pane {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background: white;
            }
            
            QTabBar::tab {
                background: #f8f9fa;
                border: 2px solid #e0e0e0;
                border-bottom: none;
                border-radius: 8px 8px 0 0;
                padding: 12px 24px;
                margin-right: 2px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QTabBar::tab:selected {
                background: white;
                border-color: #3498db;
                color: #3498db;
            }
            
            QTabBar::tab:hover {
                background: #e9ecef;
            }
        """)
    
    def convert_markdown_to_html(self, text):
        """Конвертация Markdown в HTML"""
        if not MARKDOWN_AVAILABLE:
            return text
        
        try:
            # Простой HTML шаблон без сложного CSS
            html_template = """
            <html>
            <head>
                <style>
                    body {{ 
                        font-family: 'Segoe UI', Arial, sans-serif; 
                        font-size: 14px; 
                        line-height: 1.6; 
                        color: #2c3e50;
                        margin: 10px;
                        padding: 0;
                    }}
                    h1, h2, h3, h4, h5, h6 {{ 
                        color: #2c3e50; 
                        margin-top: 20px; 
                        margin-bottom: 10px; 
                        font-weight: bold;
                    }}
                    h1 {{ 
                        font-size: 24px; 
                        border-bottom: 3px solid #3498db; 
                        padding-bottom: 8px; 
                    }}
                    h2 {{ 
                        font-size: 20px; 
                        border-bottom: 2px solid #e0e0e0; 
                        padding-bottom: 5px; 
                    }}
                    h3 {{ 
                        font-size: 18px; 
                        color: #34495e;
                    }}
                    p {{ 
                        margin: 12px 0; 
                        text-align: justify;
                    }}
                    strong, b {{ 
                        color: #3498db; 
                        font-weight: bold; 
                    }}
                    em, i {{ 
                        font-style: italic; 
                        color: #7f8c8d; 
                    }}
                    code {{ 
                        background: #f8f9fa; 
                        padding: 3px 6px; 
                        border-radius: 4px; 
                        font-family: 'Courier New', monospace;
                        border: 1px solid #e9ecef;
                    }}
                    pre {{ 
                        background: #f8f9fa; 
                        padding: 15px; 
                        border-radius: 6px; 
                        overflow-x: auto;
                        border: 1px solid #e9ecef;
                        margin: 15px 0;
                    }}
                    blockquote {{ 
                        border-left: 4px solid #3498db; 
                        margin: 15px 0; 
                        padding-left: 20px; 
                        color: #7f8c8d;
                        font-style: italic;
                    }}
                    ul, ol {{ 
                        margin: 15px 0; 
                        padding-left: 25px; 
                    }}
                    li {{ 
                        margin: 8px 0; 
                    }}
                    table {{ 
                        border-collapse: collapse; 
                        width: 100%; 
                        margin: 20px 0;
                        border: 1px solid #e0e0e0;
                    }}
                    th, td {{ 
                        border: 1px solid #e0e0e0; 
                        padding: 12px; 
                        text-align: left; 
                    }}
                    th {{ 
                        background-color: #3498db; 
                        color: white; 
                        font-weight: bold;
                    }}
                    tr:nth-child(even) {{ 
                        background-color: #f8f9fa; 
                    }}
                    tr:hover {{
                        background-color: #e9ecef;
                    }}
                </style>
            </head>
            <body>
                {content}
            </body>
            </html>
            """
            
            # Конвертируем Markdown в HTML
            html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])
            
            # Вставляем в шаблон
            full_html = html_template.format(content=html_content)
            
            return full_html
            
        except Exception as e:
            logger.warning(f"Ошибка конвертации Markdown: {e}")
            # Возвращаем простой HTML если конвертация не удалась
            return f"<html><body><p>{text}</p></body></html>"
    
    def load_model_async(self):
        """Асинхронная загрузка модели"""
        self.model_loader = ModelLoaderThread()
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.model_error.connect(self.on_model_error)
        self.model_loader.start()
    
    def on_model_loaded(self, model, tokenizer):
        """Обработчик успешной загрузки модели"""
        self.model = model
        self.tokenizer = tokenizer
        self.model_loaded = True
        
        self.status_label.setText("Модель загружена успешно")
        self.generate_btn.setEnabled(True)
    
    def on_model_error(self, error_msg):
        """Обработчик ошибки загрузки модели"""
        self.status_label.setText(f"Ошибка загрузки модели: {error_msg}")
    
    def init_hybrid_processor(self):
        """Инициализация гибридного процессора"""
        try:
            provider = self.selected_provider.lower()
            self.hybrid_processor = create_hybrid_processor(provider=provider)
            logger.info(f"Гибридный процессор инициализирован с провайдером: {provider}")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать гибридный процессор: {e}")
            self.hybrid_processor = None
            self.hybrid_checkbox.setChecked(False)
    
    def on_provider_change(self, provider):
        """Обработчик изменения провайдера"""
        self.selected_provider = provider.lower()
        logger.info(f"Выбран провайдер: {self.selected_provider}")
        
        try:
            self.hybrid_processor = create_hybrid_processor(provider=self.selected_provider)
            logger.info(f"Гибридный процессор переинициализирован с провайдером: {self.selected_provider}")
        except Exception as e:
            logger.error(f"Ошибка переинициализации процессора: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось инициализировать {self.selected_provider}: {e}")
    
    def toggle_hybrid_mode(self, enabled):
        """Переключение гибридного режима"""
        self.hybrid_enabled = enabled
        if self.hybrid_enabled and not self.hybrid_processor:
            QMessageBox.warning(self, "Предупреждение", 
                              f"Гибридный процессор недоступен. Проверьте настройки {self.selected_provider} API.")
            self.hybrid_checkbox.setChecked(False)
            self.hybrid_enabled = False
    
    def on_mode_change(self, mode):
        """Обработчик изменения режима обработки"""
        mode_mapping = {
            "Полировка": "polish",
            "Расширение": "enhance", 
            "Проверка": "verify"
        }
        
        mode_descriptions = {
            "Полировка": "Исправление ошибок и улучшение стиля",
            "Расширение": "Добавление недостающих элементов", 
            "Проверка": "Проверка на ошибки с комментариями"
        }
        
        self.mode_desc_label.setText(mode_descriptions.get(mode, ""))
        self.selected_mode = mode_mapping.get(mode, "polish")
    
    def load_input_file(self):
        """Загрузка входного файла"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с фактическими обстоятельствами",
            "", "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.input_text.setPlainText(content)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {e}")
    
    def clear_input(self):
        """Очистка входного текста"""
        self.input_text.clear()
    
    def clear_all_outputs(self):
        """Очистка всех выходных текстовых полей"""
        self.local_output_text.clear()
        self.hybrid_output_text.clear()
    
    def generate_reasoning(self):
        """Генерация документа с гибридной обработкой"""
        if not self.model_loaded:
            QMessageBox.critical(self, "Ошибка", "Модель не загружена")
            return
        
        facts = self.input_text.toPlainText().strip()
        if not facts:
            QMessageBox.warning(self, "Предупреждение", "Введите фактические обстоятельства дела")
            return
        
        # Запуск генерации в отдельном потоке
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.processing_status.setText("Начинаем генерацию...")
        
        self.generation_thread = GenerationThread(
            model=self.model,
            tokenizer=self.tokenizer,
            facts=facts,
            hybrid_processor=self.hybrid_processor if self.hybrid_enabled else None,
            mode=self.selected_mode
        )
        self.generation_thread.generation_complete.connect(self.on_generation_complete)
        self.generation_thread.generation_error.connect(self.on_generation_error)
        self.generation_thread.progress_update.connect(self.on_progress_update)
        self.generation_thread.start()
    
    def on_progress_update(self, progress, status):
        """Обновление прогресса"""
        self.progress_bar.setValue(progress)
        self.processing_status.setText(status)
    
    def on_generation_complete(self, local_response, hybrid_response):
        """Обработчик завершения генерации"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        
        # Улучшаем форматирование юридических документов
        local_formatted = self.format_legal_document(local_response)
        hybrid_formatted = self.format_legal_document(hybrid_response)
        
        # Отображаем результаты с поддержкой Markdown
        local_html = self.convert_markdown_to_html(local_formatted)
        hybrid_html = self.convert_markdown_to_html(hybrid_formatted)
        
        self.local_output_text.setHtml(local_html)
        self.hybrid_output_text.setHtml(hybrid_html)
        
        provider_name = self.selected_provider.upper()
        if self.hybrid_enabled and self.hybrid_processor:
            self.processing_status.setText(f"Готово! {provider_name} обработка завершена.")
        else:
            self.processing_status.setText("Готово! Только локальная модель QVikhr.")
    
    def format_legal_document(self, text):
        """Форматирование юридического документа для лучшего отображения"""
        if not text:
            return text
        
        # Заменяем двойные звездочки на правильный Markdown
        text = text.replace('**', '**')
        
        # Добавляем заголовки для основных разделов
        text = text.replace('В СУДЕ', '## В СУДЕ')
        text = text.replace('ИСТЕЦ:', '### ИСТЕЦ:')
        text = text.replace('ОТВЕТЧИК:', '### ОТВЕТЧИК:')
        text = text.replace('ДЕЛО №:', '### ДЕЛО №:')
        text = text.replace('Дата:', '### Дата:')
        text = text.replace('ИСКОВОЕ ЗАЯВЛЕНИЕ', '## ИСКОВОЕ ЗАЯВЛЕНИЕ')
        text = text.replace('Вводная часть', '### Вводная часть')
        text = text.replace('Мотивировочная часть', '### Мотивировочная часть')
        text = text.replace('Резолютивная часть', '### Резолютивная часть')
        text = text.replace('РЕШИЛ:', '## РЕШИЛ:')
        text = text.replace('Судья:', '### Судья:')
        
        # Добавляем разрывы строк для лучшей читаемости
        text = text.replace('. ', '.\n\n')
        text = text.replace('; ', ';\n\n')
        
        # Форматируем списки
        text = text.replace('1.', '\n1.')
        text = text.replace('2.', '\n2.')
        text = text.replace('3.', '\n3.')
        
        return text
    
    def on_generation_error(self, error_msg):
        """Обработчик ошибки генерации"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.processing_status.setText("Ошибка генерации!")
        QMessageBox.critical(self, "Ошибка генерации", f"Не удалось сгенерировать документ: {error_msg}")
    
    def save_output_file(self, source_type):
        """Сохранение результата в файл"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить документ",
            "", "Текстовые файлы (*.txt);;HTML файлы (*.html);;Все файлы (*.*)"
        )
        if filename:
            try:
                if source_type == "local":
                    content = self.local_output_text.toPlainText()
                    html_content = self.local_output_text.toHtml()
                elif source_type == "hybrid":
                    content = self.hybrid_output_text.toPlainText()
                    html_content = self.hybrid_output_text.toHtml()
                
                if filename.lower().endswith('.html'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                QMessageBox.information(self, "Успех", f"Документ сохранен из {source_type} модели")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

def main():
    app = QApplication(sys.argv)
    
    # Установка иконки приложения
    app.setWindowIcon(QIcon())
    
    # Создание и отображение главного окна
    window = LegalAssistantGUI()
    window.show()
    
    # Запуск приложения
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
