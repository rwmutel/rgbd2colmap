#!/usr/bin/env python3
import os
import subprocess
import time
import glob
import argparse
import wandb
from datetime import datetime
from pathlib import Path
import logging
import re

logger = logging.getLogger()


def run_command(command, cwd=None, measure_memory=False):
    """Run a command and return its output along with memory usage if requested"""
    logger.info(f"Running: {command}")
    
    if measure_memory:
        # Use /usr/bin/time -v to measure memory usage
        time_command = "/usr/bin/time -v"
        full_command = f"{time_command} {command}"
    else:
        full_command = command
    
    try:
        result = subprocess.run(
            full_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        
        if result.returncode != 0:
            logger.error(f"Error executing command: {command}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Command failed with return code {result.returncode}")
        
        # If we measured memory, extract the peak memory usage from stderr
        peak_memory_kb = None
        if measure_memory:
            # Look for the "Maximum resident set size" line in stderr
            match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", result.stderr)
            if match:
                peak_memory_kb = int(match.group(1))
                peak_memory_mb = peak_memory_kb / 1024
                logger.info(f"Peak memory usage: {peak_memory_mb:.2f} MB")
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "peak_memory_kb": peak_memory_kb
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {command}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise RuntimeError(f"Command failed with return code {e.returncode}")


def run_colmap_reconstruction(scene_path):
    """Run the COLMAP reconstruction pipeline on a scene"""
    scene_path = Path(scene_path)
    colmap_dir = scene_path / "colmap" / "sparse"
    
    colmap_dir.mkdir(exist_ok=True, parents=True)
    
    start_time = time.time()
    
    memory_usage = {}
    
    # Step 1: Feature extraction
    feature_result = run_command(
        f"colmap feature_extractor --database_path colmap/sparse/database.db --image_path images --ImageReader.camera_model PINHOLE",
        cwd=scene_path,
        measure_memory=True
    )
    memory_usage["feature_extractor_memory_kb"] = feature_result["peak_memory_kb"]
    
    # Step 2: Matching
    matching_result = run_command(
        f"colmap exhaustive_matcher --database_path colmap/sparse/database.db --ExhaustiveMatching.block_size 20",
        cwd=scene_path,
        measure_memory=True
    )
    memory_usage["exhaustive_matcher_memory_kb"] = matching_result["peak_memory_kb"]
    
    # Step 3: Mapping
    mapping_result = run_command(
        f"colmap mapper --database_path colmap/sparse/database.db --image_path images --output_path colmap/sparse",
        cwd=scene_path,
        measure_memory=True
    )
    memory_usage["mapper_memory_kb"] = mapping_result["peak_memory_kb"]
    
    # Step 4: Check if we need to find the largest submodel
    submodels = list(colmap_dir.glob("[0-9]"))
    if len(submodels) > 1:
        logger.warning(f"Found multiple submodels: {len(submodels)}, selecting the largest one")
        
        # Find the largest submodel by checking the number of images (using images.txt)
        largest_submodel = None
        max_images = 0
        
        for submodel in submodels:
            images_file = submodel / "images.txt"
            if images_file.exists():
                # Count the number of actual image entries in the file
                # (skipping the header lines that start with '#')
                with open(images_file, 'r') as f:
                    image_count = sum(1 for line in f if not line.startswith('#') and line.strip())
                    # Each image has 2 lines in the file, so divide by 2
                    image_count = image_count // 2
                    
                if image_count > max_images:
                    max_images = image_count
                    largest_submodel = submodel
        
        if largest_submodel and largest_submodel.name != "0":
            logger.warning(f"Selected submodel {largest_submodel.name} with {max_images} images")
            
            # Make backup of submodel 0 if it exists
            if (colmap_dir / "0").exists():
                backup_path = colmap_dir / "0_backup"
                os.makedirs(backup_path, exist_ok=True)
                run_command(
                    f"mv {str(colmap_dir)}/0/* {str(backup_path)}/",
                    cwd=scene_path
                )
            
            # Move largest submodel to replace submodel 0
            run_command(
                f"mv {str(largest_submodel)}/* {str(colmap_dir)}/0/",
                cwd=scene_path
            )
    
    bundle_result = run_command(
        f"colmap bundle_adjuster --input_path colmap/sparse/0 --output_path colmap/sparse/0",
        cwd=scene_path
    )
    elapsed_time = time.time() - start_time

    memory_usage["bundle_adjuster_memory_kb"] = bundle_result["peak_memory_kb"]
    logger.info("Bundle adjustment completed")
    
    peak_memory_values = [v for v in memory_usage.values() if v is not None]
    peak_memory_kb = max(peak_memory_values) if peak_memory_values else None

    
    return {
        "reconstruction_elapsed_time": elapsed_time,
        "submodels_count": len(submodels),
        "reconstruction_peak_memory_kb": peak_memory_kb
    }


def main():
    parser = argparse.ArgumentParser(description='Run COLMAP reconstruction on multiple scenes')
    parser.add_argument('--project', type=str, default='thesis', help='WandB project name')
    parser.add_argument('--entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--pipeline', type=str, help='Pipeline name to group by in weights and biases')
    args = parser.parse_args()
    
    # List of scenes to process
    scenes = [
        "/root/rgbd2colmap/data/MuSHRoom/classroom/iphone/short_capture/",
        # "/path/to/scene2",
        # "/path/to/scene3",
        # Add more scenes as needed
    ]
    
    for scene_path in scenes:
        scene_name = Path(scene_path).parent.parent.name
        
        run = wandb.init(
            project=args.project,
            name=f"{scene_name}-colmap-recon-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            reinit=True,
            config={
                "scene": scene_name,
                "pipeline": args.pipeline,
            }
        )
        
        # Export WANDB_RUN_ID as environment variable
        os.environ["WANDB_RUN_ID"] = run.id
        logger.info(f"WANDB_RUN_ID={run.id}")
        
        try:
            result = run_colmap_reconstruction(scene_path)
            
            wandb.log(result)
            logger.info(f"Completed COLMAP reconstruction for {scene_name} in {result['colmap_elapsed_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing scene {scene_name}: {e}")
            wandb.log({"error": str(e)})
        
        wandb.finish()


if __name__ == "__main__":
    main()