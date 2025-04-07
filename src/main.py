from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
from pipeline import RGBDReconstruction, get_cameras, get_images, get_depths

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


@hydra.main(
    config_path="../configs",
    config_name="main",
    version_base="1.3"
)
def main(cfg: DictConfig) -> None:

    # Print the configuration
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    cameras = get_cameras(cfg.reconstruction.camera_parser)
    logger.info(f"Successfully loaded {len(cameras)} cameras")
    images = get_images(cfg.reconstruction.image_parser)
    logger.info(f"Successfully loaded {len(images)} images")
    depths = get_depths(cfg.reconstruction.depth_parser)
    logger.info(f"Successfully loaded {len(depths)} depth maps")
    if not len(cameras) == len(images) == len(depths):
        logger.warning(f"Number of cameras ({len(cameras)}),"
                       f"images ({len(images)}), "
                       f"and depth maps ({len(depths)}) do not match.")
    reconstruction = RGBDReconstruction(
        cameras,
        images,
        depths,
        **cfg.reconstruction)
    logger.info(f"Successfully initialized RGBDReconstruction."
                "Starting reconstruction...")
    reconstruction.reconstruct()
    logger.info("Reconstruction finished.")
    reconstruction.visualize()

if __name__ == "__main__":
    main()