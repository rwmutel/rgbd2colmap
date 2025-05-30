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
        super().__init__(source_path)
        self.reconstruction_path = self.source_path.parent.parent

    def parse(self, path: Path = None, skip_n: int = 1) -> Dict[int, Image]:
        '''
        Parses images related to ARKit reconstruction from frames folder
        '''
        if not path:
            path = self.source_path
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(path.glob(f"*.{ext}"))
        if not image_files:
            logger.warning(f"No images found in {path}")
            return {}
        image_files = sorted(image_files)[::skip_n]
        images = {}
        for image_path in image_files:
            image_id = int(image_path.stem.split('_')[-1]) if '_' in image_path.stem else int(image_path.stem)
            image_np = cv2.imread(str(image_path))
            if image_np is None:
                logger.warning(f"Image not found or error loading {image_path}")
                continue
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            images[image_id] = Image(path=image_path, image_np=image_np)
        return images
