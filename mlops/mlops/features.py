import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def features_for_normalize():
    transforms_stats = v2.Compose([
        torchvision.transforms.RandomResizedCrop(224),  # Случайное кадрирование до 224x224
        torchvision.transforms.RandomHorizontalFlip(), # Случайное горизонтальное отражение
        torchvision.transforms.ToTensor()             # Преобразование изображения в тензор
    ])

    stats_dataset = ImageFolder(root="./animal_data/animal_dataset/train", transform=transforms_stats)

    imgs = [item[0] for item in stats_dataset]
    imgs = torch.stack(imgs, dim=0).numpy()

    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print(f"Means R, G, B: {mean_r,mean_g,mean_b}")

    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print(f"Std R, G, B: {std_r,std_g,std_b}")

    return mean_r,mean_g,mean_b, std_r,std_g,std_b

def create_transforms(mean_r,mean_g,mean_b, std_r,std_g,std_b):
    "Создание трансформеров для обучающей и тренировочной выборок."
    transforms_train = v2.Compose([
        torchvision.transforms.RandomResizedCrop(224),  # Случайное кадрирование до 224x224
        torchvision.transforms.RandomHorizontalFlip(), # Случайное горизонтальное отражение
        torchvision.transforms.ToTensor(),             # Преобразование изображения в тензор
        v2.Normalize(mean=[mean_r,mean_g,mean_b], std=[std_r,std_g,std_b]) # Нормализация с вычисленными средними и отклонениями
    ])

    transforms_test = v2.Compose([
        torchvision.transforms.RandomResizedCrop(224),  # Случайное кадрирование до 224x224
        torchvision.transforms.RandomHorizontalFlip(), # Случайное горизонтальное отражение
        torchvision.transforms.ToTensor(),             # Преобразование изображения в тензор
        v2.Normalize(mean=[mean_r,mean_g,mean_b], std=[std_r,std_g,std_b]) # Нормализация с вычисленными средними и отклонениями
    ])

    return transforms_train, transforms_test

def create_dataloader(transforms_train, transforms_test):
    "Функция создания загрузкчиков данных."
    train_dataset = ImageFolder(
        root='./animal_data/animal_dataset/train',
        transform=transforms_train
    )

    test_dataset = ImageFolder(
        root='./animal_data/animal_dataset/test',
        transform=transforms_test
    )

    BATCH_SIZE = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader