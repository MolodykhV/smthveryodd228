from pathlib import Path
from typing import List

from models import DetectionModel

model = DetectionModel(weights_path='./weights/best.pt')


def _glob_images(folder: Path, exts: List[str] = ('*.jpg', '*.png',)) -> List[Path]:
    images = []
    for ext in exts:
        images += list(folder.glob(ext))
    return images


def save_txt(self, txt_file):
    """
    Save predictions into txt file.

    Args:
        txt_file (str): txt file path.
        save_conf (bool): save confidence score or not.
    """
    boxes = self.boxes
    probs = self.probs
    texts = []
    if probs is not None:
        # Classify
        [texts.append(f'{probs.data[j]:.2f} {self.names[j]}') for j in probs.top5]
    elif boxes:
        # Detect/segment/pose
        for j, d in enumerate(boxes):
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            line = (c, conf, *d.xyxy.view(-1))
            texts.append(('%g ' * len(line)).rstrip() % line)

    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, 'a') as f:
            f.writelines(text + '\n' for text in texts)


def write_results(input_folder: str, output_folder: str):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    images = _glob_images(input_folder)
    for image in images:
        results = model.predict(image=str(image))
        output_path = str(output_folder) + '/' + str(image.with_suffix('.txt').name)
        save_txt(results[0], output_path)


def main():
    input_folder = './private/images'
    output_folder = './output/'
    write_results(input_folder, output_folder)


if __name__ == '__main__':
    main()
