from pathlib import Path
import numpy as np
from typing import Dict, Any, Iterable, Tuple, Annotated

Depth = Annotated[np.ndarray, "HxW array of depth values in millimeters"]


class DepthsParser:
    '''
    Base class for depth parsers.

    Subclasses must implement parse() method reading depth data
    from a source path with possibility of custom keyword arguments.

    Depths values must be in milimeters for compatibility with open3d.

    Depths are stored in a dictionary and must have unique ids for
    further matching with cameras and images.
    '''

    def __init__(self, source_path: Path, **kwargs: Dict[str, Any]):
        self.depths = self.parse(source_path, **kwargs)

    def parse(
        self,
        path: Path,
        **kwargs: Dict[str, Any]
    ) -> Dict[str | int, Depth]:
        '''
        Parses values from a source path.
        
        Args:
            path (Path): Path to the source data.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        Returns:
            Dict[str | int, Depth]: Dictionary of depths with unique ids.
        '''
        raise NotImplementedError(
            "parse() method must be implemented in Depths subclasses")

    def __iter__(self) -> Iterable[Tuple[str | int, Depth]]:
        return iter(self.depths.items())

    def __len__(self) -> int:
        return len(self.depths)

    def __getitem__(self, index: int | str) -> Depth:
        return self.depths[index]
