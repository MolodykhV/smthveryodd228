from PIL import Image, ImageOps, ImageEnhance
from ultralytics import YOLO


class DetectionModel:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def predict(self, image: str):
        im = Image.open(image)
        im = ImageOps.grayscale(im)
        im = ImageOps.autocontrast(im)
        im = ImageEnhance.Contrast(im).enhance(2.5)
        im = ImageEnhance.Sharpness(im).enhance(5)
        return self.model.predict(im)
