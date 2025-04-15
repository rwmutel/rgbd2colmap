# rgbd2colmap v0.2.0

This is a support repository for Roman Mutel bachelor thesis on Accelerating 3D Gaussian Splatting via RGBD-Guided Point Cloud Initialization.

Main code contribution is an extendable and configurable pipeline to convert posed RGBD images to COLMAP-compatible sparse point clouds with cameras, images and points3D files in both text and binary formats for further Gaussian Splatting optimization on captured data.

## Setup

If your goal is just to run the conversion pipeline, clone the repository with no submodules:

```shell
git clone https://github.com/rwmutel/rgbd2colmap.git
cd rgbd2colmap
```

Otherwise, to replicate batch experiments with Gaussian Splatting as the final step, clone the repository with all submodules:

```shell
git clone https://github.com/rwmutel/rgbd2colmap.git --recurse-submodules
cd rgbd2colmap
```

The repository uses Python 3.10 and a minimum set of dependencies, listed in `requirements.txt`

```shell
conda create -n rgbd2colmap python=3.10
conda activate rgbd2colmap
pip install -r requirements.txt
```

If you want to use batch processing scripts that involvle Gaussian Splatting, we recommend creating a separate conda environment due to mismatch in Python version and need for heavy packages. Refer to original Gaussian Splatting repository [README.md](./third-party/gaussian-splatting/README.md).

```shell
cd third-party/gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting
```

## Usage

Main script is `src/main.py` configured with a YAML file for [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) configuration system. Base config is `configs/main.yaml`, which can be overriden using hydra syntax. We also provide a basic config for running on MuSHRoom dataset in `configs/main_mushroom.yaml`.

Config example is as follows:

```yaml
# configs/main.yaml
reconstruction:
  parameters:
    voxel_downsample_size: 0.05
    icp_registration:
      max_iterations: 50
      relative_rmse: 1e-6
    skip_n: 1
    max_depth: 5.0
  camera_parser:
    name: ARKitCameraParser
    source_path: ./data/It-Jim/scene_example/scan_output/camera_poses.json
  image_parser:
    name: ARKitImageParser
    source_path: ./data/It-Jim/scene_example/frames
  depth_parser:
    name: ARKitDepthParser
    source_path: ./data/It-Jim/scene_example/frames

output_dir: ./data/It-Jim/scene_example/rgbd_recon/
save_reconstruction: true
save_format: bin
visualize: false
```

Thus, to experiment and **override** parameters one can run:

```shell
python src/main.py reconstruction.parameters.icp_registration.max_iterations=100 reconstruction.parameters.skip_n=3
```
To turn visualizations on, add `visualize=true` to the command. To save text files for debugging, add `save_format=text` to the command.

For easier use of **MuSHRoom parsers** change base config:

```shell
python src/main.py --config-name main_mushroom
```

with the same syntax for overriding parameters.

## Dataset

### MuSHRoom dataset

**X. Ren, W. Wang, D. Cai, T. Tuominen, J. Kannala, and E. Rahtu, ‘MuSHRoom: Multi-Sensor Hybrid Room Dataset for Joint 3D Reconstruction and Novel View Synthesis’, arXiv [cs.CV]. 2023.**

[**link**](https://zenodo.org/communities/mushroom/records?q=&l=list&p=1&s=10&sort=newest)

This particular dataset was chosen because:

1. It is modern, simplistic, and smaller compared to ScanNet
2. It contains **RGB images** with **camera poses** and **depth maps** instead of just RGB images (as DeepBlending or MipNeRF360 datasets) or images + fused point cloud (as Tanks&Temples dataset), allowing to tune and benchmark the reconstruction pipeline
3. Depth is captured by **multiple sensors**, ranging from Kinect and **iPhone** to professional Faro scanner, allowing to test the hypothesis with different classes of scanners (consumer vs professional)
4. It has proven to be a useful dataset for **Gaussian Splatting research**, shown by *DN-Splatter* (**M. Turkulainen, X. Ren, I. Melekhov, O. Seiskari, E. Rahtu, and J. Kannala, ‘DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing’, arXiv [cs.CV]. 2024.**)

Concise dataset documentation and structure:

```
MuSHRoom/
└── room_datasets/
    └── activity/  # a subfolder for a scene
        ...
    └── classroom/
        └── kinect/  # data captured with Kinect sensor
            ...
        └── iphone/  # data captured with iPhone Pro and Polycam software
            ├── long_capture/   # larger frame sequence (meant for training, 1k+ frames)
            └── short_capture/  # smaller frame sequence (meant for testing, 200-400 frames)
                ├── depth/  # depth maps stored as 16-bit 1-channel PNG (depth in millimeters)
                ├── images/  # RGB images stored as 8-bit 3-channel PNG
                ├── transformations.json    # Polycam SLAM camera poses  in ARKit coordinates (y-up, z-forward)
                └── transformations_colmap.json  # COLMAP output camera poses
    └── coffee_room/
        ...
    ...
```

### Custom Dataset

This is a custom dataset captured with consumer-grade LiDAR on iPhone Pro series. It presents the results of Apple native ARKit data obtained in real-time when scanning, captured with a simple iOS logger. The camera poses are obtained from the ARKit visual-intertial SLAM.

[**link coming soon**]()

Dataset structure and Documentation:

``` 
It-Jim/
└── home/
    ...
└── office/
    └── frames/   # folder with .jpg RGB images and .txt 255x191 (original capturing resolution) " "-separated depth values in meters
    └── scan_output/
        ├── camera_poses.json  # ARKit camera poses in y-up, z-forward coordinates matched with depth and RGB
        ├── classification.txt  # Classification of objects in the scene
        ├── mesh.obj    # Scene mesh in OBJ format
        ├── mesh.mtl    # Corresponding scene mesh materials
        ├── room.json   # Apple Roomplan output
        ├── room.plist   # Room plan in Apple proprietary format
        ├── room.usdz   # Room plan in USDZ format

        └── transformations_colmap.json  # COLMAP output camera poses
```

## Extension Guide

You are welcome to extend the pipeline with your own parsers and reconstruction optimizations!

Parsers are extended by inheriting from `CameraParser`, `ImageParser` or `DepthParser` classes, which read the data from `cfg.source_path` and return dictionaries with matching keys and values of `Camera`, `Image` and `Depth` respectively. The parsers are then registered **manually** in their parent directory `__init__.py` file:

```python
IMAGE_PARSERS = {
    "ARKitImageParser": ARKitImageParser,
    "MushroomImageParser": MushroomImageParser,
    # Add other image parsers here as needed
}
def get_image_parser(cfg: DictConfig) -> ImageParser:
    if cfg.name in IMAGE_PARSERS:
        return IMAGE_PARSERS[cfg.name](cfg.source_path)
    else:
        raise ValueError(f"Unknown image parser: {cfg.name}")
```

### Camera Conventions

Cameras have to be converted to COLMAP coordinate system and be compatible with Gaussian Splatting COLMAP parser, that is:

1. Extrinsics are inverted (camera to world transformation)
2. Y-axis is down, Z-axis is forward, X-axis is right
3. Camera type is `PINHOLE` or `SIMPLE_PINHOLE`

### Depth Conventions

DepthParser is expected to return depth as a **numpy array** of **shape (H, W)** with **float** depth values in **meters**.

DepthParser is usually **not responsible for resizing the depth maps**, as it posses no information about image size and target image size in reconstruction pipeline. **The resizing is done in the reconstruction pipeline**, where the depth maps are resized to match the RGB images or the target size.

### Present Optimizations

We decided to not unify the optimizations in the reconstruction pipeline, thus adding new optimizations is slightly complicated. Refer to existing icp_registration and voxel_downsample_size optimizations in `src/reconstruction/rgbd_reconstruction.py` for inspiration.
