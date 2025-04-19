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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def run_command(
    command,
    cwd=None,
    measure_memory=False,
    stream_output=False,
    env=None):
    """Run a command and return its output along with memory usage if requested"""
    logger.info(f"Running: {command}")
    
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    if measure_memory:
        # Use /usr/bin/time -v (GNU time utility) to measure memory usage
        time_command = "/usr/bin/time -v"
        full_command = f"{time_command} {command}"
    else:
        full_command = command
    
    try:
        process = subprocess.run(
            full_command,
            shell=True,
            stdout=subprocess.PIPE if measure_memory else None,
            stderr=subprocess.PIPE if measure_memory else None,
            text=True,
            cwd=cwd,
            env=merged_env
        )

        return_code = process.returncode
        if return_code != 0:
            logger.error(f"Error executing command: {command}")
            raise RuntimeError(f"Command failed with return code {return_code}")
        
        peak_memory_kb = None
        if measure_memory:
            # Look for the "Maximum resident set size" line in stderr
            match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", process.stderr)
            if match:
                peak_memory_kb = int(match.group(1))
                peak_memory_mb = peak_memory_kb / 1024
                logger.info(f"Peak memory usage: {peak_memory_mb:.2f} MB")
        
        return {
            "peak_memory_kb": peak_memory_kb
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {command}")
        logger.error(f"STDOUT: {e.stdout if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
        raise RuntimeError(f"Command failed with return code {e.returncode}")

def run_gaussian_splatting(scene_path, output_dir, wandb_project, run_id):
    """Run Gaussian Splatting training on a scene"""
    scene_path = Path(scene_path)
    scene_name = scene_path.parent.parent.name
    
    start_time = time.time()
    
    gs_result = run_command(
        # f"WANDB_DIR=/root/rgbd2colmap/wandb WANDB_RESUME=must WANDB_RUN_ID={run_id} " 
        "python train.py "
        f"-s {str(scene_path)}/{output_dir} "
        f"-i {str(scene_path)}/images "
        f"-m {output_dir} "
        f"--wandb_project {wandb_project} "
        f"--existing_run_id {run_id} ",
        measure_memory=False,
        stream_output=False,
        cwd="third-party/gaussian-splatting/"
    )
    
    elapsed_time = time.time() - start_time
    
    return {
        "gs_training_elapsed_time": elapsed_time,
        "gs_peak_memory_kb": gs_result["peak_memory_kb"]
    }
    

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
            points3d_file = submodel / "points3D.bin"
            if points3d_file.exists():
            # Get the size of points3D.bin in bytes
                points3d_size = points3d_file.stat().st_size
            
                if points3d_size > max_images:  # Reusing max_images variable to store max size
                    max_images = points3d_size
                    largest_submodel = submodel
        
        if largest_submodel and largest_submodel.name != "0":
            logger.warning(f"Selected submodel {largest_submodel.name} with size of points3D.vin of {max_images}")
            
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
    parser.add_argument('--pipeline', type=str, help='Pipeline name to group by in weights and biases',
                        choices=["colmap_reconstruction",
                                 "rgbd2colmap",
                                 "rgbd2colmap_colmap_poses"])
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
            name=f"{scene_name}-{args.pipeline}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
            if args.pipeline == "colmap_reconstruction":
                output_dir = "colmap"
                result = run_colmap_reconstruction(scene_path)
            elif args.pipeline == "rgbd2colmap_colmap_poses":
                output_dir = "rgbd2colmap_colmap_poses"
                start = time.time()
                result = run_command(
                    "conda run -n rgbd2colmap python src/main.py "
                    "--config-name main_mushroom "
                    f"output_dir={scene_path}/{output_dir} "
                    "reconstruction.parameters.icp_registration=False "
                    f"reconstruction.camera_parser.source_path={scene_path}transformations_colmap.json "
                    f"reconstruction.depth_parser.source_path={scene_path}depth "
                    f"reconstruction.image_parser.source_path={scene_path}images ",
                    measure_memory=True,
                    stream_output=True
                )
                elapsed_time = time.time() - start
                result.update({
                    "reconstruction_elapsed_time": elapsed_time,
                    "submodels_count": 1,
                    "reconstruction_peak_memory_kb": result["peak_memory_kb"]
                })
            elif args.pipeline == "rgbd2colmap":
                output_dir = "rgbd2colmap"
                start = time.time()
                result = run_command(
                    "conda run -n rgbd2colmap python src/main.py "
                    "--config-name main_mushroom "
                    f"output_dir={scene_path}/{output_dir} "
                    f"reconstruction.camera_parser.source_path={scene_path}transformations.json "
                    f"reconstruction.depth_parser.source_path={scene_path}depth "
                    f"reconstruction.image_parser.source_path={scene_path}images ",
                    measure_memory=True,
                    stream_output=True
                )
                elapsed_time = time.time() - start
                result.update({
                    "reconstruction_elapsed_time": elapsed_time,
                    "submodels_count": 1,
                    "reconstruction_peak_memory_kb": result["peak_memory_kb"]
                })
            else:
                raise ValueError(f"Unknown pipeline: {args.pipeline}")
            
            wandb.log(result)
            logger.info(f"Completed reconstruction for {scene_name} in {result['reconstruction_elapsed_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing scene {scene_name}: {e}")
            wandb.log({"error": str(e)})
        
        try:
            gs_results = run_gaussian_splatting(scene_path, output_dir, args.project, run.id)
            logger.info(f"Completed Gaussian Splatting training for {scene_name} in {gs_results['gs_training_elapsed_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during Gaussian Splatting training for {scene_name}: {e}")
        
        logger.info(f"Finished processing scene: {scene_name}")


if __name__ == "__main__":
    main()