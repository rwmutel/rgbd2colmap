import json
import logging
from pathlib import Path
from typing import Dict

import cv2

from .images import Image, ImageParser

logger = logging.getLogger()


class MushroomImageParser(ImageParser):
    '''
    Images parser for MuSHRoom dataset.
    https://xuqianren.github.io/publications/MuSHRoom/
    '''
    def __init__(self, source_path: str):
        super().__init__(Path(source_path))

    def parse(self, path: Path) -> Dict[int, Image]:
        '''
        Parses images related to ARKit reconstruction from json log
        '''
        images = {}
        for image_file in sorted(path.iterdir()):
            if image_file.suffix not in [".jpg", ".png", ".jpeg"]:
                continue
            image_id = image_file.stem
            image_np = cv2.imread(str(image_file))
            if image_np is None:
                logger.warning(f"Error loading {image_np}")
                continue
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            images[image_id] = Image(path=image_file, image_np=image_np)
        return images
