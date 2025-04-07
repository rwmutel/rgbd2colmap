from .depths import DepthsParser, Depth
from pathlib import Path
from typing import Dict
import json
import numpy as np


class ARKitDepthParser(DepthsParser):
    '''
    Custom ARKit logs depth parser.
    '''
    def __init__(self, source_path: str):
        self.reconstruction_path = Path(source_path).parent.parent
        super().__init__(source_path)
    
    def parse(self, path: Path) -> Dict[int, Depth]:
        '''
        Parses and rescales ARKit depths from json log
        '''
        with open(path, 'r') as file:
            data = json.load(file)
        depths = {}
        for pose in data['poses']:
            depth_path = self.reconstruction_path / Path(pose['depth'])
            depth_id = int(Path(pose['image']).stem.split('_')[-1])
            depth_map = np.loadtxt(depth_path, delimiter=',', dtype=np.float32)
            depths[depth_id] = depth_map
        return depths
