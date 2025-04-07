from .images import ImageParser, Image
from .arkit_images import ARKitImageParser
from omegaconf import DictConfig


IMAGE_PARSERS = {
    "ARKitImageParser": ARKitImageParser,
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
