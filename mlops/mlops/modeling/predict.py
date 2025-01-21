import torch
from PIL import Image
import torchvision.transforms as T

def predict_model(mean_r, mean_g, mean_b, std_r, std_g, std_b, image_path, pretrained_model, labels_map):
    "Функция предсказывания случайного изображения"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")

    # Преобразования изображения (должны быть аналогичны используемым при обучении)
    transform = T.Compose([
        T.Resize((256, 256)),  # Размер, использованный при обучении
        T.ToTensor(),  # Преобразование в тензор
        T.Normalize(mean=[mean_r, mean_g, mean_b], std=[std_r, std_g, std_b])  # Нормализация
    ])

    # Применение преобразований
    image_tensor = transform(image).unsqueeze(0)  # Добавляем batch-измерение

    # Переводим модель в режим оценки
    pretrained_model.eval()

    # Переносим данные на устройство
    image_tensor = image_tensor.to(device)

    # Классификация изображения
    with torch.no_grad():
        output = pretrained_model(image_tensor)
        predicted_class = output.argmax(dim=1).item()

    # Вывод предсказанного класса
    print(f"Предсказанный класс: {labels_map[predicted_class]}")
