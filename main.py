import os
from pylabel import importer
from ultralytics import YOLO
import requests
from zipfile import ZipFile
import random
from sklearn.model_selection import train_test_split
import shutil
random.seed(123)


def get_data(url: str = 'https://storage.yandexcloud.net/contestfiles/A_Changellenge_Urbanhack/urbanhack-train.zip'):
    """
    Функция скачивания тестового датасета и преобразования его в нужный формат
    :param url: Ссылка на датафрейм
    :return:
    """
    # Загрузка файла
    r = requests.get(url, allow_redirects=True)
    r.raise_for_status()
    open('urbanhack_original.zip', 'wb').write(r.content)

    # Разархивирование
    with ZipFile('urbanhack_original.zip', 'r') as f:
        f.extractall('dataset_unprepared')

    # Создание папок
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    os.makedirs('dataset/labels/test', exist_ok=True)
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/images/test', exist_ok=True)

    # Преобразование аннотаций
    dataset = importer.ImportCoco(path='dataset_unprepared/urbanhack-train/annotations/instances_default.json',
                                  path_to_images='dataset_unprepared/urbanhack-train/images')
    dataset.export.ExportToYoloV5(output_path='dataset_unprepared/new_annotations')

    # Разбивка на тестовое, валидационное и тренировочные множества
    images_names = os.listdir('dataset_unprepared/new_annotations')
    train_names, test_names = train_test_split(images_names, train_size=0.9)
    train_names, val_names = train_test_split(train_names, train_size=0.6)

    # Перемещение по папкам
    for image in train_names:
        os.rename(f'dataset_unprepared/new_annotations/{image}',
                  f'dataset/labels/train/{image}')
        os.rename(f'dataset_unprepared/urbanhack-train/images/{image}'[:-3] + 'jpg',
                  f'dataset/images/train/{image}'[:-3] + 'jpg')
    for image in val_names:
        os.rename(f'dataset_unprepared/new_annotations/{image}',
                  f'dataset/labels/val/{image}')
        os.rename(f'dataset_unprepared/urbanhack-train/images/{image}'[:-3] + 'jpg',
                  f'dataset/images/val/{image}'[:-3] + 'jpg')
    for image in test_names:
        os.rename(f'dataset_unprepared/new_annotations/{image}',
                  f'dataset/labels/test/{image}')
        os.rename(f'dataset_unprepared/urbanhack-train/images/{image}'[:-3] + 'jpg',
                  f'dataset/images/test/{image}'[:-3] + 'jpg')

    # Создание yaml файла
    with open('dataset/dataset.yaml', 'wb') as f:
        f.write(bytes('path: ../dataset/\ntrain: images/train/\nval: images/val/\ntest: images/test/\nnc: 4\nnames:\n   0: none\n   1: window\n   2: empty\n   3: filled', 'utf-8'))


    # Удаление лишнего
    shutil.rmtree('dataset_unprepared')
    os.remove('urbanhack_original.zip')


def train_model():
    model = YOLO('yolov8n.yaml')
    # model = YOLO('yolov8n.pt')
    results = model.train(data='dataset/dataset.yaml', epochs=5)
    results = model.val()
    return model


def main():
    try:
        get_data()
    except requests.HTTPError:
        print('Что-то не так со ссылкой для скачивания')
    except Exception:
        print('Что-то не так')

    model = train_model()
    model.predict('dataset/images/test/0000004046building.jpg')


if __name__ == '__main__':
    main()
