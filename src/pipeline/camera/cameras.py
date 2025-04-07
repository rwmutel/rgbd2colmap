from pathlib import Path
import numpy as np
from typing import Dict, Any, Iterable, Tuple
from dataclasses import dataclass
import open3d as o3d

@dataclass
class Camera:
    intrinsic: np.ndarray  # 3x3 matrix
    extrinsic: np.ndarray  # 4x4 camera to world matrix

    def get_o3d_intrinsic(
        self,
        width: int,
        height: int
    ) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=self.intrinsic[0, 0],
            fy=self.intrinsic[1, 1],
            cx=self.intrinsic[0, 2],
            cy=self.intrinsic[1, 2],
        )



class CameraParser:
    '''
    Base class for camera parsers.

    Subclasses must implement parse() method reading camera data
    from a source path with possibility of custom keyword arguments.

    Cameras' extrinsics are 4x4 camera to world matrices.

    Cameras are stored in a dictionary and must have unique ids for
    further matching with images and depths.
    '''

    def __init__(self, source_path: Path, **kwargs: Dict[str, Any]):
        self.cameras = self.parse(source_path, **kwargs)

    @staticmethod
    def parse(path: Path, **kwargs: Dict[str, Any]) -> Dict[str | int, Camera]:
        '''
        Parses values from a source path.
        
        Args:
            path (Path): Path to the source data.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        Returns:
            Dict[str | int, Camera]: Dictionary of cameras with unique ids.
        '''
        raise NotImplementedError(
            "parse() method must be implemented in Cameras subclasses")

    def __iter__(self) -> Iterable[Tuple[str | int, Camera]]:
        return iter(self.cameras.items())

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, index: int | str) -> Camera:
        return self.cameras[index]
