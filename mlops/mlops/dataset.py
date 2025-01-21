from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


from duckduckgo_search import DDGS
import os
import requests

def load_links(class_list):
    "Загрузка ссылок к изображениям"
    for animal in class_list:
        with DDGS() as ddgs:
            ddgs_images_gen = ddgs.images(
            f'{animal} photo',
            region="wt-wt",
            size="Medium",
            type_image="photo",
            max_results=1000,
            )
            with open(f'animal_data/{animal}.txt', 'w', encoding='utf-8') as f:  # пишем в файл полученные ссылки на изображения для скачивания
                for r in ddgs_images_gen:
                    f.write(f"{r['image']}\n")

def load_image(class_list, output_dir):
    "Функция загрузки изображений указанных классов по ссылкам."
    # Убедитесь, что папка существует
    os.makedirs(output_dir, exist_ok=True)

    # Цикл по всем файлам из списка
    for animal in class_list:
        target_folder = os.path.join(output_dir, animal)  # Папка для конкретного животного
        os.makedirs(target_folder, exist_ok=True)  # Создаём папку, если её нет

        # Открываем файл с ссылками
        with open(f'animal_data/{animal}.txt', "r") as file:
            links = file.readlines()  # Читаем все строки (ссылки)

        # Цикл по каждой ссылке
        for i, link in enumerate(links, start=1):
            link = link.strip()  # Удаляем лишние пробелы и символы перевода строки
            if not link:
                continue  # Пропускаем пустые строки

            try:
                # Загружаем изображение
                response = requests.get(link, stream=True, timeout=10)  # Stream=True для больших файлов
                response.raise_for_status()  # Проверяем на ошибки HTTP

                # Определяем имя файла
                filename = os.path.join(target_folder, f"image_{i}.jpg")

                # Сохраняем файл
                with open(filename, "wb") as img_file:
                    for chunk in response.iter_content(chunk_size=8192):  # Сохраняем по частям
                        img_file.write(chunk)

                print(f"Скачано: {filename}")
            except requests.exceptions.RequestException as e:
                print(f"Ошибка при загрузке {link}: {e}")
