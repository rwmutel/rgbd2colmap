from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import open3d as o3d


@dataclass
class Camera:
    intrinsic: np.ndarray  # 3x3 matrix
    extrinsic: np.ndarray  # 4x4 camera to world matrix
    _width: int | None = None
    _height: int | None = None

    def get_o3d_intrinsic(
        self,
        width: int = None,
        height: int = None,
    ) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            width=width if width is not None else self.width,
            height=height if height is not None else self.height,
            fx=self.intrinsic[0, 0],
            fy=self.intrinsic[1, 1],
            cx=self.intrinsic[0, 2],
            cy=self.intrinsic[1, 2],
        )

    def qvec(self) -> np.ndarray:
        '''
        Inspired by https://github.com/colmap/colmap/blob/f4a7e143a17b7a6b8a86406b82f60f3a2767c602/scripts/python/read_write_model.py#L546
        '''
        Rxx, Ryx, Rzx, \
        Rxy, Ryy, Rzy, \
        Rxz, Ryz, Rzz = self.extrinsic[:3, :3].flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec
    
    def tvec(self) -> np.ndarray:
        return self.extrinsic[:3, 3]

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, width: int):
        if width != self._width:
            self.intrinsic[0, 2] = width / 2
        self._width = width

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, height: int):
        if height != self._height:
            self.intrinsic[1, 2] = height / 2
        self._height = height


class CameraParser:
    '''
    Base class for camera parsers.

    Subclasses must implement parse() method reading camera data
    from a source path with possibility of custom keyword arguments.

    Cameras' extrinsics are 4x4 camera to world matrices.

    Cameras are stored in a dictionary and must have unique ids for
    further matching with images and depths.
    '''

    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        self.cameras = {}

    @staticmethod
    def parse(path: Path, skip_n: int = 1) -> Dict[str | int, Camera]:
        '''
        Parses values from a source path.
        Must assign self.cameras: Dict[str | int, Camera] to the parsed values.

        Args:
            path (Path): Path to the source data.
            skip_n (int): Stride to iterate over cameras ([::skip_n]). Default is 1.
        Returns:
            Dict[str | int, Camera]: Dictionary of cameras with unique ids.
        '''
        raise NotImplementedError(
            "parse() method must be implemented in Cameras subclasses")

    def __iter__(self) -> Iterable[Tuple[str | int, Camera]]:
        if not self.cameras:
            raise ValueError("Cameras are not parsed yet.")
        return iter(self.cameras.items())

    def __len__(self) -> int:
        if not self.cameras:
            raise ValueError("Cameras are not parsed yet.")
        return len(self.cameras)

    def __getitem__(self, index: int | str) -> Camera:
        if not self.cameras:
            raise ValueError("Cameras are not parsed yet.")
        return self.cameras[index]
