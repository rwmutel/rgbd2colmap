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
        self.reconstruction_path = Path(source_path).parent.parent
        super().__init__(Path(source_path))

    def parse(self, path: Path) -> Dict[int, Image]:
        '''
        Parses images related to ARKit reconstruction from json log
        '''
        images = {}
        for image_file in path.iterdir():
            if image_file.suffix not in [".jpg", ".png", ".jpeg"]:
                continue
            image_id = image_file.stem
            image = cv2.imread(str(image_file))
            if image is None:
                logger.warning(f"Error loading {str(image_file)}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[image_id] = image
        return images
