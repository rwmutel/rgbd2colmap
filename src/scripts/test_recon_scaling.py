#!/usr/bin/env python3
import os
import time
import argparse
import wandb
from datetime import datetime
from pathlib import Path
import logging
import shutil

from src.scripts.run_mushroom_colmap_recon import run_colmap_reconstruction, run_command

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description='Run COLMAP reconstruction with varying number of frames')
    parser.add_argument('--project', type=str, default='thesis', help='WandB project name')
    parser.add_argument('--scene', type=str, required=True, help='Path to the scene to process')
    args = parser.parse_args()
    
    # Define the frame counts to experiment with
    frame_counts = [100, 250, 500, 750]
    
    # Pipelines to run
    pipelines = ["colmap_reconstruction", "rgbd2colmap_no_icp"]
    # pipelines = ["colmap_reconstruction"]
    
    scene_path = args.scene
    scene_name = Path(scene_path).parent.parent.name
    
    temp_base_dir = Path(f"{scene_path}/frame_experiments")
    temp_base_dir.mkdir(exist_ok=True)
    
    original_images_path = Path(f"{scene_path}/images")
    original_depth_path = Path(f"{scene_path}/depth")
    
    original_images = sorted(list(original_images_path.glob("*.jpg")) + list(original_images_path.glob("*.png")))
    original_depth = sorted(list(original_depth_path.glob("*.png")))
    
    max_frame_count = max(frame_counts)
    if len(original_images) < max_frame_count:
        logger.warning(f"Only {len(original_images)} images available, less than maximum requested {max_frame_count}")
        frame_counts = [fc for fc in frame_counts if fc <= len(original_images)]
    
    for frame_count in frame_counts:
        logger.info(f"Processing with {frame_count} frames")
        
        temp_dir = temp_base_dir / f"frames_{frame_count}"
        temp_images_dir = temp_dir / "images"
        temp_depth_dir = temp_dir / "depth"
        
        temp_dir.mkdir(exist_ok=True)
        temp_images_dir.mkdir(exist_ok=True)
        temp_depth_dir.mkdir(exist_ok=True)
        
        shutil.copy(f"{scene_path}/transformations.json", temp_dir / "transformations.json")
        
        # Sample first N frames
        selected_images = original_images[:frame_count]
        selected_depth = original_depth[:frame_count]
        
        for img_path in selected_images:
            shutil.copy(img_path, temp_images_dir / img_path.name)
        for depth_path in selected_depth:
            shutil.copy(depth_path, temp_depth_dir / depth_path.name)
        
        for pipeline in pipelines:
            run = wandb.init(
                project=args.project,
                name=f"{scene_name}-{pipeline}-frames{frame_count}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                reinit=True,
                config={
                    "scene": scene_name,
                    "pipeline": pipeline,
                    "frame_count": frame_count,
                    "experiment": "scaling",
                }
            )
            
            logger.info(f"WANDB_RUN_ID={run.id}")
            
            try:
                if pipeline == "colmap_reconstruction":
                    output_dir = f"colmap_frames{frame_count}"
                    start = time.time()
                    result = run_colmap_reconstruction(temp_dir, output_dir=output_dir)
                    print(result)
                    elapsed_time = time.time() - start
                    
                elif pipeline == "rgbd2colmap_no_icp":
                    output_dir = f"rgbd2colmap_frames{frame_count}"
                    start = time.time()
                    result = run_command(
                        "conda run -n rgbd2colmap python src/main.py "
                        "--config-name main_mushroom "
                        f"output_dir={temp_dir}/{output_dir} "
                        "reconstruction.parameters.icp_registration=False "
                        f"reconstruction.camera_parser.source_path={temp_dir}/transformations.json "
                        f"reconstruction.depth_parser.source_path={temp_dir}/depth "
                        f"reconstruction.image_parser.source_path={temp_dir}/images ",
                        measure_memory=True,
                    )
                    elapsed_time = time.time() - start
                    result.update({
                        "reconstruction_elapsed_time": elapsed_time,
                        "submodels_count": 1,
                        "reconstruction_peak_memory_kb": result["peak_memory_kb"]
                    })

                result["frame_count"] = frame_count

                wandb.log(result)
                logger.info(f"Completed {pipeline} for {scene_name} with {frame_count} frames in {elapsed_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Error processing {pipeline} for {scene_name} with {frame_count} frames: {e}")
                wandb.log({"error": str(e), "frame_count": frame_count})
            
            run.finish()
    
    # shutil.rmtree(temp_base_dir)
    logger.info(f"Experiment completed for scene {scene_name} with frame counts {frame_counts}")

if __name__ == "__main__":
    main()