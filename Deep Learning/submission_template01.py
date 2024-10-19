import numpy as np
import json
import re

import torch

# Сохраняем версию torch в переменную version
version = torch.__version__

# не изменяйте код в блоке ниже! Он нужен для проверки правильности вашего кода.
# __________start of block__________
import re
assert version is not None, 'Версия PyTorch не сохранилась в переменную version'
major_version, minor_version = re.findall("\d+\.\d+", version)[0].split('.')
assert float(major_version) >= 2 or (float(major_version) >= 1 and float(minor_version) >= 7), 'Нужно обновить PyTorch'
# __________end of block__________

import torch
import torch.nn as nn

def create_model():
    # Используем nn.Sequential для создания модели
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),  # Первый линейный слой 784 -> 256
        nn.ReLU(),                       # Функция активации ReLU
        nn.Linear(256, 16, bias=True),   # Второй линейный слой 256 -> 16
        nn.ReLU(),                       # Функция активации ReLU
        nn.Linear(16, 10, bias=True)     # Последний линейный слой 16 -> 10 (без активации)
    )

    return model

# Создаем модель
model = create_model()

# не изменяйте код в блоке ниже! Он нужен для проверки правильности вашего кода.
# __________start of block__________
for param in model.parameters():
    nn.init.constant_(param, 1.)

assert torch.allclose(model(torch.ones((1, 784))), torch.ones((1, 10)) * 3215377.), 'Что-то не так со структурой модели'
# __________end of block__________

import torch
import torch.nn as nn

def count_parameters(model):
    # Подсчитываем количество параметров в модели
    return sum(param.numel() for param in model.parameters())

# не изменяйте код в блоке ниже! Он нужен для проверки правильности вашего кода.
# __________start of block__________
small_model = nn.Linear(128, 256)
assert count_parameters(small_model) == 128 * 256 + 256, 'Что-то не так, количество параметров неверное'

medium_model = nn.Sequential(*[nn.Linear(128, 32, bias=False), nn.ReLU(), nn.Linear(32, 10, bias=False)])
assert count_parameters(medium_model) == 128 * 32 + 32 * 10, 'Что-то не так, количество параметров неверное'
print("Seems fine!")
# __________end of block__________
