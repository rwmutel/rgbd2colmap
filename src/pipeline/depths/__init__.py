from omegaconf import DictConfig

from .arkit_depths import ARKitDepthParser
from .depths import DepthsParser

DEPTH_PARSERS = {
    "ARKitDepthParser": ARKitDepthParser,
    # Add other depth parsers here as needed
}


def get_depth_parser(cfg: DictConfig) -> DepthsParser:
    '''
    Factory function for creating depth parsers based on the configuration.
    '''
    if cfg.name in DEPTH_PARSERS:
        return DEPTH_PARSERS[cfg.name](cfg.source_path)
    else:
        raise ValueError(f"Unknown depth parser: {cfg.name}")
