#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "Vikhrmodels/QVikhr-3-4B-Instruction"  # можно заменить на свою модель
BATCH_SIZE = 2
SEQ_LEN = 512
NUM_BATCHES = 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Генерация фейковых данных
    dummy_inputs = ["Тестовая строка для загрузки GPU."] * BATCH_SIZE
    inputs = tokenizer(dummy_inputs, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Старт теста: {NUM_BATCHES} батчей × {BATCH_SIZE} seq_len={SEQ_LEN}")
    with torch.no_grad():
        for _ in tqdm(range(NUM_BATCHES), desc="Прогон батчей"):
            outputs = model(**inputs)

    print("✅ Тест завершён")

if __name__ == "__main__":
    main()
