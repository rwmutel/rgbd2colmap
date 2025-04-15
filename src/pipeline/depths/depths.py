from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, Tuple

import numpy as np

Depth = Annotated[np.ndarray, "HxW array of depth values in millimeters"]


class DepthsParser:
    '''
    Base class for depth parsers.

    Subclasses must implement parse() method reading depth data
    from a source path with possibility of custom keyword arguments.

    Depths values must be in meters for compatibility with open3d.

    Depths are stored in a dictionary and must have unique ids for
    further matching with cameras and images.
    '''

    def __init__(self, source_path: Path, **kwargs: Dict[str, Any]):
        self.source_path = Path(source_path)
        self.depths = {}

    def parse(
        self,
        path: Path,
        skip_n: int = 1,
    ) -> Dict[str | int, Depth]:
        '''
        Parses values from a source path with optional stride [::skip_n].
        
        Args:
            path (Path): Path to the source data.
            skip_n (int): Stride to iterate over depths ([::skip_n]). Default is 1.
        Returns:
            Dict[str | int, Depth]: Dictionary of depths with unique ids.
        '''
        raise NotImplementedError(
            "parse() method must be implemented in Depths subclasses")

    def __iter__(self) -> Iterable[Tuple[str | int, Depth]]:
        if not self.depths:
            raise ValueError("Depths are not parsed yet.")
        return iter(self.depths.items())

    def __len__(self) -> int:
        if not self.depths:
            raise ValueError("Depths are not parsed yet.")
        return len(self.depths)

    def __getitem__(self, index: int | str) -> Depth:
        if not self.depths:
            raise ValueError("Depths are not parsed yet.")
        return self.depths[index]
