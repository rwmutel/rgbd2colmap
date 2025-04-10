import json
import logging
from pathlib import Path
from typing import Dict

import cv2

from .images import Image, ImageParser

logger = logging.getLogger()


class ARKitImageParser(ImageParser):
    '''
    Custom ARKit logs image parser.
    '''
    def __init__(self, source_path: str):
        self.reconstruction_path = Path(source_path).parent.parent
        super().__init__(source_path)

    def parse(self, path: Path) -> Dict[int, Image]:
        '''
        Parses images related to ARKit reconstruction from json log
        '''
        with open(path, 'r') as file:
            data = json.load(file)
        images = {}
        for pose in data['poses']:
            image_path = self.reconstruction_path / Path(pose['image'])
            image_id = int(Path(pose['image']).stem.split('_')[-1])
            image_np = cv2.imread(str(image_path))
            if image_np is None:
                logger.warning(f"Image not found or error loading {image_path}")
                continue
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            images[image_id] = Image(path=image_path, image_np=image_np)
        return images
