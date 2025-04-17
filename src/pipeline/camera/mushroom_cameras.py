import json
from pathlib import Path
from typing import Dict

import numpy as np

from .cameras import Camera, CameraParser

# same as ARKit fix. MuSHRoom uses polycam for iphone capture
# https://github.com/TUTvision/MuSHRoom/blob/main/README.md#data-structure
# https://github.com/PolyCam/polyform?tab=readme-ov-file#cameras
# https://developer.apple.com/documentation/arkit/arconfiguration/worldalignment-swift.enum/gravity
POLYCAM_FIX = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


class MushroomCameraParser(CameraParser):
    '''
    Camera parser for MuSHRoom dataset.
    https://xuqianren.github.io/publications/MuSHRoom/
    '''
    def __init__(self, source_path: str):
        super().__init__(source_path)

    def parse(self, path: Path = None, skip_n: int = 1) -> Dict[int, Camera]:
        '''
        Parses ARKit cameras from json log
        '''
        if not path:
            path = self.source_path
        with open(path, 'r') as file:
            data = json.load(file)
        if "colmap" in path.stem:
            transforms_format = "colmap"
            fl_x, fl_y = data["fl_x"], data["fl_y"]
            cx, cy = data["cx"], data["cy"]

        cameras = {}
        for pose in data['frames']:
            camera_id = Path(pose['file_path']).stem
            if transforms_format != "colmap":
                fl_x, fl_y = pose["fl_x"], pose["fl_y"]
                cx, cy = pose["cx"], pose["cy"]
            intrinsics = np.array([
                [fl_x, 0, cx],
                [0, fl_y, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            extrinsics = np.array(pose['transform_matrix'])
            # if transforms_format != "colmap":
            extrinsics = extrinsics @ POLYCAM_FIX
            extrinsics = np.linalg.inv(extrinsics)
            cameras[camera_id] = Camera(intrinsics, extrinsics)
        # sort cameras by id to have adjacent frames
        cameras = dict(sorted(cameras.items(),
                              key=lambda item: int(item[0].split("_")[-1])))
        return cameras
