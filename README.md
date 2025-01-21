# MLOps

# mlops

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Этот проект представляет ключевые аспекты процесса MLOps:

Организация структуры кода для задач машинного обучения.
Экспорт модели в формат ONNX для повышения ее совместимости и производительности.

## Установка

1. **Клонирование репозитория**:
Клонируйте проект и перейдите в соответствующую директорию:

   ```bash
   git https://github.com/Nenneke2999/MLOps.git
   cd mlops
   ```

2. **Создание виртуального окружения (необязательно)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate  # Windows
   ```

3. **Установка зависимостей**:
   Убедитесь, что у вас установлен Python версии 3.8 или выше.

   ```bash
   pip install -r requirements.txt
   ```

## Project Organization

```
├── Makefile           <- Удобные команды, например, `make data` или `make train`.
├── README.md          <- Основная документация для пользователей проекта.
├── models             <- Папка с сохраненными моделями.
├── notebooks          <- Jupyter-ноутбуки с экспериментами.
│   ├── animal_data        <- Данные для обучения и тестирования.
│   ├── Fine_tune_Mironova.ipynb    <- Ноутбук для дообучения модели.
│   └── lite.ipynb      <- Оптимизированный ноутбук под текущий проект.
├── mlops   <- Исходный код проекта.
│   ├── dataset.py         <- Код для загрузки и обработки данных.
│   ├── features.py        <- Генерация признаков для модели.
│   ├── modeling           <- Код для обучения и инференса модели.
│   │   ├── train.py       <- Скрипт для обучения модели.
│   │   └── predict.py     <- Код для выполнения предсказаний.
├── requirements.txt   <- Список всех зависимостей.
└── README.md          <- Текущий файл описания проекта.
```

--------