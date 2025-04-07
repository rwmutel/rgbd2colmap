from .cameras import CameraParser, Camera
from typing import Dict
from pathlib import Path
import json
import numpy as np

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
        self.reconstruction_path = Path(source_path).parent.parent
        super().__init__(source_path)
    
    def parse(self, path: Path) -> Dict[int, Camera]:
        '''
        Parses ARKit cameras from json log
        '''
        with open(path, 'r') as file:
            data = json.load(file)
        cameras = {}
        for pose in data['poses']:
            camera_id = int(Path(pose['image']).stem.split('_')[-1])
            intrinsics = np.array(pose['intrinsic']).reshape(3, 3)
            extrinsics = np.array(pose['transform']).reshape(4, 4).T @ ARKIT_FIX
            extrinsics[3, 3] = 1.0
            extrinsics = np.linalg.inv(extrinsics)
            cameras[camera_id] = Camera(intrinsics, extrinsics)
        return cameras
