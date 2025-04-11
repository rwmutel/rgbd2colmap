from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


@dataclass
class Image:
    '''
    Image class representing a single image with its id and path.

    Attributes:
        path (Path | str): Path to the image file.
        image (np.ndarray): HxWxC array of RGB values in 0-255 range
    '''
    path: Path | str
    image_np: np.ndarray


class ImageParser:
    '''
    Base class for image parsers.

    Subclasses must implement parse() method reading image data
    from a source path with possibility of custom keyword arguments.

    Images are stored in a dictionary and must have unique ids for
    further matching with cameras and depths.
    '''

    def __init__(self, source_path: Path, **kwargs: Dict[str, Any]):
        self.images = self.parse(source_path, **kwargs)

    def parse(
        self,
        path: Path,
        **kwargs: Dict[str, Any]
    ) -> Dict[str | int, Image]:
        '''
        Parses values from a source path.

        Args:
            path (Path): Path to the source data.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        Returns:
            Dict[str | int, Image]: Dictionary of images with unique ids.
        '''
        raise NotImplementedError(
            "parse() method must be implemented in Images subclasses")

    def __iter__(self) -> Iterable[Tuple[str | int, Image]]:
        return iter(self.images.items())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int | str) -> Image:
        return self.images[index]
