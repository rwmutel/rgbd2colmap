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
        process = subprocess.Popen(
            full_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=merged_env
        )
        
        stdout_data, stderr_data = "", ""
        
        if stream_output:
            # Stream output while capturing it
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                if not stdout_line and not stderr_line and process.poll() is not None:
                    break
                
                if stdout_line:
                    stdout_line = stdout_line.rstrip()
                    logger.info(stdout_line)
                    stdout_data += stdout_line + "\n"
                
                if stderr_line:
                    stderr_line = stderr_line.rstrip()
                    logger.error(stderr_line)
                    stderr_data += stderr_line + "\n"
        else:
            stdout_data, stderr_data = process.communicate()
        
        return_code = process.poll() if stream_output else process.wait()
        if return_code != 0:
            logger.error(f"Error executing command: {command}")
            logger.error(f"STDOUT: {stdout_data}")
            logger.error(f"STDERR: {stderr_data}")
            raise RuntimeError(f"Command failed with return code {return_code}")
        
        peak_memory_kb = None
        if measure_memory:
            # Look for the "Maximum resident set size" line in stderr
            match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr_data)
            if match:
                peak_memory_kb = int(match.group(1))
                peak_memory_mb = peak_memory_kb / 1024
                logger.info(f"Peak memory usage: {peak_memory_mb:.2f} MB")
        
        return {
            "stdout": stdout_data,
            "stderr": stderr_data,
            "peak_memory_kb": peak_memory_kb
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {command}")
        logger.error(f"STDOUT: {e.stdout if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"STDERR: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
        raise RuntimeError(f"Command failed with return code {e.returncode}")

def run_gaussian_splatting(scene_path, wandb_project, run_id):
    """Run Gaussian Splatting training on a scene"""
    scene_path = Path(scene_path)
    scene_name = scene_path.parent.parent.name
    
    output_dir = f"output/{scene_name}_colmap"
    start_time = time.time()
    
    gs_result = run_command(
        f"conda run -n gaussian_splatting python train.py "
        f"-s {str(scene_path)}/colmap "
        f"-i {str(scene_path)}/images "
        f"-m {output_dir} "
        f"--wandb_project {wandb_project} "
        f"--existing_run_id {run_id}",
        cwd="third-party/gaussian-splatting",
        measure_memory=True,
        stream_output=True,
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
            logger.info(f"Completed COLMAP reconstruction for {scene_name} in {result['reconstruction_elapsed_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing scene {scene_name}: {e}")
            wandb.log({"error": str(e)})
        
        try:
            gs_results = run_gaussian_splatting(scene_path, args.project, run.id)
            logger.info(f"Completed Gaussian Splatting training for {scene_name} in {gs_results['gs_training_elapsed_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during Gaussian Splatting training for {scene_name}: {e}")
        
        logger.info(f"Finished processing scene: {scene_name}")


if __name__ == "__main__":
    main()