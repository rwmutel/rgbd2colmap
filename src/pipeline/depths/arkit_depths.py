import json
from pathlib import Path
from typing import Dict

import numpy as np

from .depths import Depth, DepthsParser


class ARKitDepthParser(DepthsParser):
    '''
    Custom ARKit logs depth parser.
    '''
    def __init__(self, source_path: str):
        self.reconstruction_path = Path(source_path).parent.parent
        super().__init__(source_path)

    def parse(self, path: Path) -> Dict[int, Depth]:
        '''
        Parses and rescales ARKit depths (depth*_*.txt) from a source path.
        '''
        depths = {}
        for depth_path in path.glob("*.txt"):
            depth_id = int(depth_path.stem.split('_')[-1])
            depth_map = np.loadtxt(depth_path, delimiter=',', dtype=np.float32)
            depths[depth_id] = depth_map
        return depths
