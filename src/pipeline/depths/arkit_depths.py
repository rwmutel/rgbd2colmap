import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .depths import Depth, DepthsParser

logger = logging.getLogger()


class ARKitDepthParser(DepthsParser):
    '''
    Custom ARKit logs depth parser.
    '''
    def __init__(self, source_path: str):
        super().__init__(source_path)
        self.reconstruction_path = self.source_path.parent.parent

    def parse(self, path: Path = None, skip_n: int = 1) -> Dict[int, Depth]:
        '''
        Parses and rescales ARKit depths (depth*_*.txt) from a source path.
        '''
        if not path:
            path = self.source_path
        depths = {}
        depth_paths = sorted(self.source_path.glob("depth_*.txt"))[::skip_n]
        for depth_path in depth_paths:
            depth_id = int(depth_path.stem.split('_')[-1])
            try:
                depth_map = np.loadtxt(depth_path, delimiter=',', dtype=np.float32)
                depths[depth_id] = depth_map
            except ValueError:
                logger.warning(f"Failed to load depth map from {depth_path}. Skipping.")
                continue
        return depths
