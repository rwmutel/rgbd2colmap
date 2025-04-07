from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
from pipeline import RGBDReconstruction, get_camera_parser, get_image_parser, get_depth_parser
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


@hydra.main(
    config_path="../configs",
    config_name="main",
    version_base="1.3"
)
def main(cfg: DictConfig) -> None:

    # Print the configuration
    # logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    cameras = get_camera_parser(cfg.reconstruction.camera_parser).cameras
    logger.info(f"Successfully loaded {len(cameras)} cameras")
    images = get_image_parser(cfg.reconstruction.image_parser).images
    logger.info(f"Successfully loaded {len(images)} images")
    depths = get_depth_parser(cfg.reconstruction.depth_parser).depths
    logger.info(f"Successfully loaded {len(depths)} depth maps")
    if not len(cameras) == len(images) == len(depths):
        logger.warning(f"Number of cameras ({len(cameras)}),"
                       f"images ({len(images)}), "
                       f"and depth maps ({len(depths)}) do not match.")
    reconstruction = RGBDReconstruction(cameras, images, depths)
    logger.info("Successfully initialized RGBDReconstruction.")
    logger.info("Starting reconstruction...")
    start = time.time()
    reconstruction.reconstruct()
    end = time.time()
    logger.info(f"Reconstruction took {end - start:.2f} seconds.")
    # reconstruction.visualize()


if __name__ == "__main__":
    main()
