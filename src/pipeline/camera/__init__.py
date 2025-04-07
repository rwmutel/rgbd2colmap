from omegaconf import DictConfig

from .arkit_cameras import ARKitCameraParser
from .cameras import Camera, CameraParser  # noqa: F401

CAMERA_PARSERS = {
    "ARKitCameraParser": ARKitCameraParser,
    # Add other camera parsers here as needed
}


def get_camera_parser(cfg: DictConfig) -> CameraParser:
    '''
    Factory function for creating camera parsers based on the configuration.
    '''
    if cfg.name in CAMERA_PARSERS:
        return CAMERA_PARSERS[cfg.name](cfg.source_path)
    else:
        raise ValueError(f"Unknown camera parser: {cfg.name}")
