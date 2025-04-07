from .cameras import CameraParser, Camera
from .arkit_cameras import ARKitCameraParser
from omegaconf import DictConfig

CAMERA_PARSERS = {
    "ARKitCameraParser": ARKitCameraParser,
    # Add other camera parsers here as needed
}


def get_cameras(cfg: DictConfig) -> CameraParser:
    '''
    Factory function for creating camera parsers based on the configuration.
    '''
    if cfg.name in CAMERA_PARSERS:
        return CAMERA_PARSERS[cfg.name](cfg.source_path, **cfg.kwargs)
    else:
        raise ValueError(f"Unknown camera parser: {cfg.name}")
