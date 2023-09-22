import logging
import os
import shutil
import warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFilter, ImageOps
from numba import prange
from pylabel import importer
from sklearn.model_selection import train_test_split
from ultralytics import YOLO, settings

warnings.filterwarnings('ignore')
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='logs.log', format=formatter, level=logging.INFO)


def decolor_all_images(test_img_path: str = 'dataset/images/test/',
                       train_img_path: str = 'dataset/images/train/',
                       val_img_path: str = 'dataset/images/val/'):
    """
    Убираем цвет на фото и выравниваем контраст
    :param test_img_path: пусть к тестовым изображениям
    :param train_img_path: путь к тренировочным изображениям
    :param val_img_path: путь к валидационным изображениям
    :return:
    """
    for path in [test_img_path, train_img_path, val_img_path]:
        names: list = os.listdir(path)
        for image in prange(len(names)):
            image_extention = os.path.splitext(names[image])[1]
            im = Image.open(path + names[image])
            im = ImageOps.grayscale(im)
            im = ImageOps.autocontrast(im)
            os.remove(path + names[image])
            im.save(path + names[image][:-len(image_extention)] + '.png', 'png', quality=1)


def work_on_photos(img_path: str = 'dataset/images/train/', lab_path: str = 'dataset/labels/train/'):
    """
    Расширение датафрейма за счет обработки фото
    :param img_path: путь до изображений
    :param lab_path: путь до лейблов
    :return:
    """
    img_names: list = os.listdir(img_path)
    for image in img_names:
        image_extention = os.path.splitext(image)[1]
        label: pd.DataFrame = pd.read_csv(lab_path + image[:-len(image_extention)] + '.txt',
                                          sep=' ', header=None, index_col=None)
        im = Image.open(img_path + image)

        im_distorbed = np.array(im)
        A = im_distorbed.shape[0] / 100.0
        w = 5 / im_distorbed.shape[1]
        shift = lambda x: A * np.sin(3 * np.pi * x * w)
        for i in range(im_distorbed.shape[1]):
            im_distorbed[:, i] = np.roll(im_distorbed[:, i], int(shift(i)))
        im_distorbed = Image.fromarray(im_distorbed)
        im_distorbed.save(img_path + image[:-len(image_extention)] + '_distorbed' + image_extention, quality=1)
        label.to_csv(lab_path + image[:-len(image_extention)] + '_distorbed.txt',
                     sep=' ', header=None, index=False)

        im_blurred = im.filter(ImageFilter.BoxBlur(1.9))
        im_blurred.save(img_path + image[:-len(image_extention)] + '_blurred' + image_extention, quality=1)
        label.to_csv(lab_path + image[:-len(image_extention)] + '_blurred.txt',
                     sep=' ', header=None, index=False)

        label[1] = 1 - label[1]
        label.to_csv(lab_path + image[:-len(image_extention)] + '_inversed.txt',
                     sep=' ', header=None, index=False)
        im_invert = im.transpose(Image.FLIP_LEFT_RIGHT)
        im_invert.save(img_path + image[:-len(image_extention)] + '_inversed' + image_extention, quality=1)


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
    train_names, test_names = train_test_split(images_names, train_size=0.95)
    train_names, val_names = train_test_split(train_names, train_size=0.7)

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
        f.write(bytes(
            'train: images/train/\nval: images/val/\ntest: images/test/\nnc: 4\nnames:\n   0: none\n   1: window\n   2: empty\n   3: filled',
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
    except Exception:
        print('Что-то не так')
    settings.reset()
    settings.update({'runs_dir': 'runs', 'weights_dir': 'weights', 'datasets_dir': ''})
    decolor_all_images()
    work_on_photos()
    model, results = train_model()

    # prediction = model('dataset/images/test/0000004046building.jpg')
    print(results.box.map50)
    # print(prediction)


if __name__ == '__main__':
    main()
