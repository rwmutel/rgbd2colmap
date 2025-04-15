import json
from pathlib import Path
from typing import Dict

import numpy as np

from .cameras import Camera, CameraParser

ARKIT_FIX = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


class ARKitCameraParser(CameraParser):
    '''
    Custom ARKit logs camera parser.
    '''
    def __init__(self, source_path: str):
        super().__init__(source_path)
        self.reconstruction_path = self.source_path.parent.parent

    def parse(self, path: Path = None, skip_n: int = 1) -> Dict[int, Camera]:
        '''
        Parses ARKit cameras from json log
        '''
        if not path:
            path = self.source_path
        with open(path, 'r') as file:
            data = json.load(file)
        cameras = {}
        strided_sorted_poses = sorted(
            data['poses'],
            key=lambda x: int(Path(x['image']).stem.split('_')[-1])
            )[::skip_n]
        for pose in strided_sorted_poses:
            camera_id = int(Path(pose['image']).stem.split('_')[-1])
            intrinsics = np.array(pose['intrinsic']).reshape(3, 3)
            extrinsics = np.array(pose['transform']).reshape(4, 4).T @ ARKIT_FIX
            extrinsics[3, 3] = 1.0
            extrinsics = np.linalg.inv(extrinsics)
            cameras[camera_id] = Camera(intrinsics, extrinsics)
        return cameras
