# rgbd2colmap v1.0.0

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

**Hint:** if you would like to use newer cuda version, you should look into `third-party/gaussian-splatting/patches`, which contains [patches from community](https://github.com/graphdeco-inria/gaussian-splatting/issues/923).
To apply the patches, run:

```shell
cd third-party/gaussian-splatting/SIBR_viewers
git apply ../patches/SIBR_viewers.patch.txt
cd ../submodules/simple-knn
git apply ../../patches/simple-knn.patch.txt
```

## Reproducing Experimental Results

Results on MuSHRoom dataset are obtained via runner script that integrates both reconstruction and Gaussian Splatting training. Options for reconstruction are

+ colmap_reconstruction
+ rgbd2colmap
+ rgbd2colmap_colmap_poses

Keep in mind that the script is rather hardcoded, so you should adapt `scenes` list and 3D GS `train.py` script path for reproducing.

Example command for running the script:

```shell
python src/scripts/run_mushroom_colmap_recon.py --project rgbd2colmap --pipeline rgbd2colmap_colmap_poses
``` 

## Usage

Main script to run reconstruction is `src/main.py` configured with a YAML file for [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) configuration system. Base config is `configs/main.yaml`, which can be overriden using hydra syntax:

+ `reconstruction.parameters.icp_registration.max_iterations=100` for parameters defined in the config
+ `+reconstruction.parameters.target_image_size="[480,640]"` for adding new parameters
+ `-reconstruction.parameters.voxel_downsample_size` for removing parameters

We provide a basic config for running on MuSHRoom dataset in `configs/main_mushroom.yaml`.

Config example is as follows:

```yaml
# configs/main.yaml
reconstruction:
  skip_n: 1
  parameters:
    # target_image_size: [640, 480]
    # target_pcd_size: 10000
    voxel_size: 0.05
    icp_registration:
      max_iterations: 50
      relative_rmse: 1e-6
    max_depth: 5.0
    remove_stat_outliers:
      nb_neighbors: 20
      std_ratio: 1.0
  camera_parser:
    name: ARKitCameraParser
    source_path: ./data/office26/scan_output/camera_poses.json
  image_parser:
    name: ARKitImageParser
    source_path: ./data/It-Jim/office26/frames
  depth_parser:
    name: ARKitDepthParser
    source_path: ./data/It-Jim/office26/frames

output_dir: ./data/It-Jim/office26/rgbd_recon/
save_reconstruction: true
save_format: bin
visualize: false
```

Thus, to experiment and **override** or **remove** parameters one can run:

```shell
python src/main.py ~reconstruction.parameters.icp_registration.max_iterations reconstruction.skip_n=3
```

To turn visualizations on, add `visualize=true` to the command. To save text files for debugging, add `save_format=text` to the command.

For easier use of **MuSHRoom parsers** change base config:

```shell
python src/main.py --config-name main_mushroom
```

## Datasets

### Custom Dataset

[**link**](https://drive.google.com/drive/folders/1m9kgqdaphVVSgP8UOTz3LbwzdPqdEh3L?usp=sharing)

**Hint:** download dataset easily with `gdown`:

```shell
pip install gdown
gdown --folder 1m9kgqdaphVVSgP8UOTz3LbwzdPqdEh3L
```

This is a custom dataset captured with iPhone 12 Pro. It presents the results of Apple native ARKit data obtained in real-time when scanning, captured with an in-house built iOS logger. The camera poses are obtained from the ARKit visual-intertial odometry.

Dataset contains captures of scenes from open-space offices in Kyiv and Kharkiv, Ukraine. Dataset is made to emulate "casual" captures with possibly rough movements and blurred images.

Frame distribution is as follows:

|    Scene   | Frames |
|:----------:|:------:|
| office26   |   133  |
| promodo    |   176  |
| conference |   385  |



Dataset structure and Documentation:

``` 
It-Jim/
└── home/
    ...
└── office/
    └── frames/   # folder with .jpg RGB images and .txt 255x191 (original capturing resolution) " "-separated depth values in meters
    └── scan_output/
        └── camera_poses.json  # ARKit camera poses in y-up, z-forward coordinates matched with depth and RGB
```

### MuSHRoom dataset

**X. Ren, W. Wang, D. Cai, T. Tuominen, J. Kannala, and E. Rahtu, ‘MuSHRoom: Multi-Sensor Hybrid Room Dataset for Joint 3D Reconstruction and Novel View Synthesis’, arXiv [cs.CV]. 2023.**

[**link**](https://zenodo.org/communities/mushroom/records?q=&l=list&p=1&s=10&sort=newest)

To put our approach on the 3D GS research map, we choose to evaluate it on an established academic dataset.

This particular dataset was chosen because:

1. It is modern, simplistic, and smaller compared to ScanNet
2. It contains **RGB images** with **camera poses** and **depth maps** instead of just RGB images (as DeepBlending or MipNeRF360 datasets) or images + fused point cloud (as Tanks&Temples dataset), allowing to tune and benchmark the reconstruction pipeline
3. Depth is captured by **multiple sensors**, ranging from Kinect and **iPhone** to professional Faro scanner, allowing to test the hypothesis with different classes of scanners (consumer vs professional)
4. It has proven to be a useful dataset for **Gaussian Splatting research**, shown by *DN-Splatter* (**M. Turkulainen, X. Ren, I. Melekhov, O. Seiskari, E. Rahtu, and J. Kannala, ‘DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing’, arXiv [cs.CV]. 2024.**)
5. In addition to iPhone captures with camera poses calculated by **Polycam** SLAM, authors provide **COLMAP** camera poses for the same captures, allowing to compare the results of different reconstruction pipelines

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

General conventions is to sort parsed entities (cameras/depth maps/images) by their id for stride downsampling ( with stride optionally defined in `reconstruction.skip_n` parameter) and further matching based on id's. 

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

Present optimizations are:
1. `voxel_downsample_size` - voxel downsampling of the point cloud
2. `remove_stat_outliers` - statistical outlier removal of the point cloud
3. `icp_registration` - Colored **I**terative **C**losest **P**oint registration of the point cloud
