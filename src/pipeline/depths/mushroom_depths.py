import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .depths import Depth, DepthsParser


class MushroomDepthParser(DepthsParser):
    '''
    Depth parser for MuSHRoom dataset.
    https://xuqianren.github.io/publications/MuSHRoom/
    '''
    def __init__(self, source_path: str):
        super().__init__(Path(source_path))

    def parse(self, path: Path) -> Dict[int, Depth]:
        '''
        Parses and rescales ARKit depths from json log
        '''
        depths = {}
        for depth_file in path.iterdir():
            if depth_file.suffix != '.png':
                continue
            depth_id = depth_file.stem
            depth_map = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            depth_map = depth_map.astype(np.float32) / 1000.0
            depths[depth_id] = depth_map
            # depth_map[depth_map == 0] = np.nan
            # depth_map[depth_map > 10] = np.nan
        return depths
