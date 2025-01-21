import timm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler


def train_model(num_classes, train_loader, test_loader):
    "Функция обучения модели."
    pretrained_model = timm.create_model('resnet50.a1_in1k', pretrained=True)

    # "Замораживаем" веса
    for param in pretrained_model.parameters():
        param.requires_grad = False

    num_classes = 10  # Количество классов в задаче

    # Заменяем "голову"
    # .fc для вашей модели может иметь другое имя
    # В nn.Sequential добавьте 1-2 скрытых слоя (nn.Linear, nn.ReLU)
    pretrained_model.fc = nn.Sequential(
        nn.Linear(pretrained_model.fc.in_features, 256),  # Промежуточный слой
        nn.ReLU(),                                       # Активация
        nn.Dropout(0.3),                                 # Dropout для регуляризации
        nn.Linear(256, num_classes)                     # Выходной слой
    )

    loss_fn = nn.CrossEntropyLoss() # Определяем функцию потерь
    optimizer = torch.optim.Adam(pretrained_model.fc.parameters(), lr=0.001) # Определяем оптимизатор

    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        # Переключаем модель в режим обучения
        pretrained_model.train()

        # Обучение
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # Перенос данных на устройство
            images, labels = images.to(device), labels.to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход
            outputs = pretrained_model(images)
            loss = loss_fn(outputs, labels)

            # Обратное распространение
            loss.backward()

            # Шаг оптимизации
            optimizer.step()

            # Накопление метрик
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Вычисление средней потери и точности для обучения
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Переключаем модель в режим оценки
        pretrained_model.eval()

        # Тестирование
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                # Перенос данных на устройство
                images, labels = images.to(device), labels.to(device)

                # Прямой проход
                outputs = pretrained_model(images)
                loss = loss_fn(outputs, labels)

                # Накопление метрик
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        # Вычисление средней потери и точности для теста
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct_test / total_test

        # Шаг планировщика
        scheduler.step()

        # Вывод метрик за эпоху
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

def fine_tune_model(pretrained_model, train_loader, test_loader):
    "Функция дообучения модели."
    n = 2
    for param in pretrained_model.layer4[-n:].parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([
        {'params': pretrained_model.layer4.parameters(), 'lr': 1e-4},  # Размороженные параметры layer4
        {'params': pretrained_model.fc.parameters(), 'lr': 1e-3},     # Размороженные параметры fc
    ], lr=1e-2, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)

    for epoch in range(num_epochs):
        # Переводим модель в режим обучения
        pretrained_model.train()

        # Обучение
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # Перенос данных на устройство
            images, labels = images.to(device), labels.to(device)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = pretrained_model(images)

            # Вычисление потерь
            loss = loss_fn(outputs, labels)

            # Обратное распространение ошибки
            loss.backward()

            # Шаг оптимизации
            optimizer.step()

            # Накопление метрик
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Вычисление средней потери и точности для обучения
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Тестирование
        pretrained_model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = pretrained_model(images)
                loss = loss_fn(outputs, labels)

                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        # Вычисление средней потери и точности для теста
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct_test / total_test

        # Шаг планировщика
        scheduler.step()

        # Вывод метрик за эпоху
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )
