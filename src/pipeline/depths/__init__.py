from .depths import DepthsParser, Depth
from omegaconf import DictConfig


DEPTH_PARSERS = {
    "DepthsParser": DepthsParser,
    # Add other depth parsers here as needed
}


def get_depths(cfg: DictConfig) -> DepthsParser:
    '''
    Factory function for creating depth parsers based on the configuration.
    '''
    if cfg.name in DEPTH_PARSERS:
        return DEPTH_PARSERS[cfg.name](cfg.source_path, **cfg.kwargs)
    else:
        raise ValueError(f"Unknown depth parser: {cfg.name}")
