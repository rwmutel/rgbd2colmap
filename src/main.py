import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from pipeline import (OutputFormat, RGBDReconstruction, get_camera_parser,
                      get_depth_parser, get_image_parser)

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
    start = time.time()
    cameras = get_camera_parser(cfg.reconstruction.camera_parser).cameras
    logger.info(f"Successfully loaded {len(cameras)} cameras")
    depths = get_depth_parser(cfg.reconstruction.depth_parser).depths
    logger.info(f"Successfully loaded {len(depths)} depth maps")
    images = get_image_parser(cfg.reconstruction.image_parser).images
    logger.info(f"Successfully loaded {len(images)} images")
    if not len(cameras) == len(images) == len(depths):
        logger.warning(f"Number of cameras ({len(cameras)}),"
                       f"images ({len(images)}), "
                       f"and depth maps ({len(depths)}) do not match.")
    reconstruction = RGBDReconstruction(
        cameras, images, depths,
        cfg.reconstruction.parameters)
    end = time.time()
    logger.info("Successfully initialized RGBDReconstruction in "
                f"{end - start:.2f} seconds.")
    logger.info("Starting reconstruction...")
    start = time.time()
    reconstruction.reconstruct()
    end = time.time()
    logger.info(f"Reconstruction took {end - start:.2f} seconds.")
    if cfg.get("visualize", False):
        logger.info("Visualizing reconstruction...")
        reconstruction.visualize()

    start = time.time()
    if cfg.get("save_reconstruction", False):
        out_format = OutputFormat(cfg.get("save_format", OutputFormat.BIN.value))
        logger.info(f"Saving reconstruction in {out_format} format...")
        if out_format == OutputFormat.TXT:
            reconstruction.save_txt(Path(cfg.output_dir))
        elif out_format == OutputFormat.BIN:
            reconstruction.save(Path(cfg.output_dir))
        else:
            logger.warning(f"Unsupported output format: {out_format}. "
                           f"Reconstruction will saved in binary format.")
            reconstruction.save(Path(cfg.output_dir))
    end = time.time()
    logger.info(f"Saving reconstruction took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
