import logging
import os
import shutil
import warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from pylabel import importer
from sklearn.model_selection import train_test_split
from ultralytics import YOLO, settings

warnings.filterwarnings('ignore')
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='logs.log', format=formatter, level=logging.INFO)


def base_photo_filter(image: Image):
    im = ImageOps.grayscale(image)
    im = ImageOps.autocontrast(im)
    im = ImageEnhance.Contrast(im).enhance(2.5)
    im = ImageEnhance.Sharpness(im).enhance(5)
    return im


def expand_images(image: Image):
    # Искаженное фото
    im_distorbed = np.array(image)
    A = im_distorbed.shape[0] / 250.0
    w = 5 / im_distorbed.shape[1]
    shift = lambda x: A * np.sin(3 * np.pi * x * w)
    for i in range(im_distorbed.shape[1]):
        im_distorbed[:, i] = np.roll(im_distorbed[:, i], int(shift(i)))
    im_distorbed = Image.fromarray(im_distorbed)

    # Заблюренное фото
    im_blurred = image.filter(ImageFilter.BoxBlur(1.9))

    # Инвертированное фото
    im_invert = image.transpose(Image.FLIP_LEFT_RIGHT)
    return im_distorbed, im_blurred, im_invert


def get_data(url: str = 'https://storage.yandexcloud.net/contestfiles/A_Changellenge_Urbanhack/urbanhack-train.zip'):
    """
    Функция скачивания тестового датасета и преобразования его в нужный формат
    :param url: Ссылка на датафрейм
    :return:
    """
    # Загрузка файла
    logging.info('loading data')
    r = requests.get(url, allow_redirects=True)
    r.raise_for_status()
    open('urbanhack_original.zip', 'wb').write(r.content)

    # Разархивирование
    with ZipFile('urbanhack_original.zip', 'r') as f:
        f.extractall('dataset_unprepared')

    # Создание папок
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    os.makedirs('test_images/labels', exist_ok=True)
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('test_images/images', exist_ok=True)

    # Преобразование аннотаций
    dataset = importer.ImportCoco(path='dataset_unprepared/urbanhack-train/annotations/instances_default.json',
                                  path_to_images='dataset_unprepared/urbanhack-train/images')
    dataset.export.ExportToYoloV5(output_path='dataset_unprepared/new_annotations')

    # Разбивка на тестовое, валидационное и тренировочные множества
    images_names = os.listdir('dataset_unprepared/new_annotations')
    train_names, test_names = train_test_split(images_names, train_size=0.95)
    train_names, val_names = train_test_split(train_names, train_size=0.7)

    # Перемещение по папкам
    for image in train_names:
        annotation = pd.read_csv(f'dataset_unprepared/new_annotations/{image}',
                                 sep=' ', header=None, index_col=False)
        annotation[0] = annotation[0] - 1
        annotation_distorbed, annotation_blurred, annotation_inverted = annotation.copy(), annotation.copy(), annotation.copy()
        annotation_inverted[1] = 1 - annotation_inverted[1]
        annotation.to_csv(f'dataset/labels/train/{image}', sep=' ', header=False, index=False)
        annotation_distorbed.to_csv(f'dataset/labels/train/{image}'[:-4] + '_distorbed.txt',
                                    sep=' ', header=False, index=False)
        annotation_inverted.to_csv(f'dataset/labels/train/{image}'[:-4] + '_inverted.txt',
                                   sep=' ', header=False, index=False)
        annotation_blurred.to_csv(f'dataset/labels/train/{image}'[:-4] + '_blurred.txt',
                                  sep=' ', header=False, index=False)
        # os.rename(f'dataset_unprepared/new_annotations/{image}',
        #          f'dataset/labels/train/{image}')
        img: Image = base_photo_filter(Image.open(f'dataset_unprepared/urbanhack-train/images/{image}'[:-3] + 'jpg'))
        img_distorbed, img_blurred, img_inverted = expand_images(img)
        img.save(f'dataset/images/train/{image}'[:-4] + '.png', format='png', quality=1)
        img_distorbed.save(f'dataset/images/train/{image}'[:-4] + '_distorbed.png', format='png', quality=1)
        img_blurred.save(f'dataset/images/train/{image}'[:-4] + '_blurred.png', format='png', quality=1)
        img_inverted.save(f'dataset/images/train/{image}'[:-4] + '_inverted.png', format='png', quality=1)

        # os.rename(f'dataset_unprepared/urbanhack-train/images/{image}'[:-3] + 'jpg',
        #          f'dataset/images/train/{image}'[:-3] + 'jpg')
    for image in val_names:
        # os.rename(f'dataset_unprepared/new_annotations/{image}',
        #          f'dataset/labels/val/{image}')
        annotation = pd.read_csv(f'dataset_unprepared/new_annotations/{image}',
                                 sep=' ', header=None, index_col=False)
        annotation[0] = annotation[0] - 1
        annotation.to_csv(f'dataset/labels/val/{image}', sep=' ', header=False, index=False)

        img: Image = base_photo_filter(Image.open(f'dataset_unprepared/urbanhack-train/images/{image}'[:-3] + 'jpg'))
        img.save(f'dataset/images/val/{image}'[:-3] + 'png', format='png', quality=1)
        # os.rename(f'dataset_unprepared/urbanhack-train/images/{image}'[:-3] + 'jpg',
        #          f'dataset/images/val/{image}'[:-3] + 'jpg')
    for image in test_names:
        # os.rename(f'dataset_unprepared/new_annotations/{image}',
        #          f'dataset/labels/test/{image}')
        os.rename(f'dataset_unprepared/urbanhack-train/images/{image}'[:-3] + 'jpg',
                  f'test_images/images/{image}'[:-3] + 'jpg')
        annotation = pd.read_csv(f'dataset_unprepared/new_annotations/{image}',
                                 sep=' ', header=None, index_col=False)
        annotation[0] = annotation[0] - 1
        annotation.to_csv(f'test_images/labels/{image}', sep=' ', header=False, index=False)

    # Создание yaml файла
    with open('dataset/dataset.yaml', 'wb') as f:
        f.write(bytes(
            'train: images/train/\nval: images/val/\ntest: images/test/\nnc: 3\nnames:\n   0: window\n   1: empty\n   2: filled',
            'utf-8'))

    # Удаление лишнего
    shutil.rmtree('dataset_unprepared')
    os.remove('urbanhack_original.zip')
    logging.info('data loaded')


def train_model():
    """
    Функция создания и обучения модели
    :return: Модель
    """
    logging.info('training model')
    # model = YOLO('yolov8l.yaml')
    model = YOLO('yolov8n.yaml')
    # model = YOLO('yolov8n.pt')
    results = model.train(data='./dataset/dataset.yaml', epochs=10)
    logging.info('validating model')
    results = model.val()
    return model, results


def main():
    try:
        get_data()
    except requests.HTTPError:
        print('Что-то не так со ссылкой для скачивания')
    settings.reset()
    settings.update({'runs_dir': 'runs', 'weights_dir': './weights', 'datasets_dir': ''})
    model, results = train_model()

    # prediction = model('dataset/images/test/0000004046building.jpg')
    print(results.box.map50)
    if os.path.exists('./weights'):
        shutil.rmtree('./weights')
    os.rename('./runs/detect/train/weights', './weights')
    shutil.rmtree('./runs/detect/train')
    # print(prediction)


if __name__ == '__main__':
    main()
