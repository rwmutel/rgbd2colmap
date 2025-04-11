import json
from pathlib import Path
from typing import Dict

import numpy as np

from .cameras import Camera, CameraParser


class MushroomCameraParser(CameraParser):
    '''
    Camera parser for MuSHRoom dataset.
    https://xuqianren.github.io/publications/MuSHRoom/
    '''
    def __init__(self, source_path: str):
        super().__init__(source_path)

    def parse(self, path: Path) -> Dict[int, Camera]:
        '''
        Parses ARKit cameras from json log
        '''
        with open(path, 'r') as file:
            data = json.load(file)
        cameras = {}
        for pose in data['frames']:
            camera_id = Path(pose['file_path']).stem
            intrinsics = np.array([
                [pose["fl_x"], 0, pose["cx"]],
                [0, pose["fl_y"], pose["cy"]],
                [0, 0, 1]
            ], dtype=np.float32)
            extrinsics = np.array(pose['transform_matrix'])
            extrinsics = np.linalg.inv(extrinsics)
            cameras[camera_id] = Camera(intrinsics, extrinsics)
        return cameras
