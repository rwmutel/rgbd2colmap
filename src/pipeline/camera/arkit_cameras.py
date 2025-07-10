import json
from pathlib import Path
from typing import Dict, List, Tuple

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
        Parses ARKit cameras from frames folder
        '''
        if not path:
            path = self.source_path
        camera_files = list(path.glob('camera*.txt'))
        if not camera_files:
            raise FileNotFoundError(f"No camera files found in {path}")
        cameras = {}
        strided_sorted_poses = sorted(
            camera_files,
            key=lambda x: int(Path(x).stem.split('_')[-1])
            )[::skip_n]
        for camera_file in strided_sorted_poses:
            camera_id = int(Path(camera_file).stem.split('_')[-1])
            
            with open(camera_file, 'r') as f:
                lines = f.readlines()
            
            extrinsics_start, intrinsics_start = self.find_pose_sections(lines)
            if extrinsics_start is None or intrinsics_start is None:
                raise ValueError(f"Missing extrinsics or intrinsics section in {camera_file}")

            extrinsics = self.parse_txt_matrix(lines[extrinsics_start:extrinsics_start + 4])
            intrinsics = self.parse_txt_matrix(lines[intrinsics_start:intrinsics_start + 3])

            extrinsics = extrinsics @ ARKIT_FIX
            extrinsics[3, 3] = 1.0
            extrinsics = np.linalg.inv(extrinsics)

            cameras[camera_id] = Camera(intrinsics, extrinsics)
        return cameras

    @staticmethod
    def parse_txt_matrix(ext_lines: List[str]) -> np.ndarray:
        extrinsics_data = []
        for line in ext_lines:
            line = line.strip()
            if line:
                row = [float(x.strip()) for x in line.split(',')]
                extrinsics_data.append(row)
        return np.array(extrinsics_data, dtype=np.float32)
    
    @staticmethod
    def find_pose_sections(lines: List[str]) -> Tuple[int, int]:
        extrinsics_start = None
        intrinsics_start = None
        for i, line in enumerate(lines):
            if line.strip() == 'extrinsics:':
                extrinsics_start = i + 1
            elif line.strip() == 'intrinsics:':
                intrinsics_start = i + 1
        return extrinsics_start, intrinsics_start