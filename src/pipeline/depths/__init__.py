from pathlib import Path

from omegaconf import DictConfig

from .arkit_depths import ARKitDepthParser
from .depths import Depth, DepthsParser  # noqa: F401
from .mushroom_depths import MushroomDepthParser

DEPTH_PARSERS = {
    "ARKitDepthParser": ARKitDepthParser,
    "MushroomDepthParser": MushroomDepthParser,
    # Add other depth parsers here as needed
}


def get_depth_parser(cfg: DictConfig) -> DepthsParser:
    '''
    Factory function for creating depth parsers based on the configuration.
    '''
    if cfg.name in DEPTH_PARSERS:
        return DEPTH_PARSERS[cfg.name](Path(cfg.source_path))
    else:
        raise ValueError(f"Unknown depth parser: {cfg.name}")
