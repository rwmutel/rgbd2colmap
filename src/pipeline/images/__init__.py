from .images import ImageParser, Image
from omegaconf import DictConfig


IMAGE_PARSERS = {
    "ImageParser": ImageParser,
    # Add other image parsers here as needed
}


def get_images(cfg: DictConfig) -> ImageParser:
    '''
    Factory function for creating image parsers based on the configuration.
    '''
    if cfg.name in IMAGE_PARSERS:
        return IMAGE_PARSERS[cfg.name](cfg.source_path, **cfg.kwargs)
    else:
        raise ValueError(f"Unknown image parser: {cfg.name}")
