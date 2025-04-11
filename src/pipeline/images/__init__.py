from omegaconf import DictConfig

from .arkit_images import ARKitImageParser
from .images import Image, ImageParser  # noqa: F401
from .mushroom_images import MushroomImageParser

IMAGE_PARSERS = {
    "ARKitImageParser": ARKitImageParser,
    "MushroomImageParser": MushroomImageParser,
    # Add other image parsers here as needed
}


def get_image_parser(cfg: DictConfig) -> ImageParser:
    '''
    Factory function for creating image parsers based on the configuration.
    '''
    if cfg.name in IMAGE_PARSERS:
        return IMAGE_PARSERS[cfg.name](cfg.source_path)
    else:
        raise ValueError(f"Unknown image parser: {cfg.name}")
