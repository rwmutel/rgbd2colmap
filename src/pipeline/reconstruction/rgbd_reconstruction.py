import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import open3d as o3d
from tqdm import tqdm

from ..camera import Camera
from ..depths import Depth
from ..images import Image
from ..utils.colmap_io import (copy_images, write_cameras_binary,
                               write_cameras_text, write_images_binary,
                               write_images_text, write_points3D_binary,
                               write_points3D_text)

logger = logging.getLogger(__name__)


class RGBDReconstruction:
    '''
    Class that agregates cameras, images and depths
    and reconstructs scene point cloud, optionally supporting
    pre- and postprocessing of parsed data.

    Implements saving of the scene into COLMAP format for
    further usage with Gaussian Splatting
    '''

    def __init__(
        self,
        cameras: Dict[str | int, Camera],
        images: Dict[str | int, Image],
        depths: Dict[str | int, Depth],
    ):
        # TODO: move into config, implement image rescaling as well as depths
        self.target_width, self.target_height = self._get_image_shape(images)
        self.cameras = self._rescale_intrinsics(cameras)
        self.images = images
        self.rgbds = self._combine_images_and_depths(images, depths)

    def _rescale_intrinsics(
        self,
        cameras: Dict[str | int, Camera]
    ) -> Dict[str | int, Camera]:
        '''
        Rescales camera intrinsics to match the target image size.
        Assumes all cameras have the same intrinsic matrix.
        '''
        for camera in cameras.values():
            if camera.width != self.target_width \
               or camera.height != self.target_height:
                camera.width = self.target_width
                camera.height = self.target_height
        return cameras

    def _get_image_shape(
        self,
        images: Dict[str | int, Image]
    ) -> Tuple[int, int]:
        '''
        Returns the shape of the first image in the dictionary.
        Assumes all images have the same shape.
        '''
        first_image = next(iter(images.values()))
        return first_image.image_np.shape[1], first_image.image_np.shape[0]

    def _combine_images_and_depths(
        self,
        images: Dict[str | int, Image],
        depths: Dict[str | int, Depth],
    ) -> Dict[str | int, o3d.geometry.RGBDImage]:
        '''
        Combines images and depths into a single dictionary
        using the same keys for both.
        '''
        rgbds = {}
        if set(images.keys()) != set(depths.keys()):
            logger.warning(
                "Images and depths do not match."
                "Only matching keys will be used.")
            logger.warning(
                f"Mismatched keys: {set(images.keys()) - set(depths.keys())}")
        for key in tqdm(images.keys(), desc="Combining images and depths"):
            if key in depths:
                depth_o3d = self._rescale_depth(depths[key])
                image_o3d = o3d.geometry.Image(images[key].image_np)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    image_o3d,
                    depth_o3d,
                    depth_scale=1.0,
                    depth_trunc=1000.0,
                    convert_rgb_to_intensity=False,
                )
                rgbds[key] = rgbd
            else:
                logger.warning(f"Depth for image {key} not found.")
        return rgbds

    def _rescale_depth(self, depth: Depth) -> o3d.geometry.Image:
        '''
        Rescales depth image to match the image shape.
        Assumes camera intrisics are given for the RGB image
        '''
        depth = cv2.resize(
            depth,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC,
        )
        depth = o3d.geometry.Image(depth)
        return depth

    def reconstruct(self) -> o3d.geometry.PointCloud:
        '''
        Reconstructs scene point cloud from cameras, images and depths.
        Optionally includes pre- and postprocessing steps.
        '''
        # self.preprocess()
        self.pcd = self._reconstruct_scene()
        # self.postprocess()
        return self.pcd

    def _reconstruct_scene(self):
        '''
        Reconstructs the scene by unprojecting depth maps from given cameras
        '''
        pcd = o3d.geometry.PointCloud()
        for key, rgbd in tqdm(self.rgbds.items(), desc="Unprojecting depth"):
            camera = self.cameras[key]
            intrinsic = camera.get_o3d_intrinsic()
            extrinsic = camera.extrinsic
            pcd_temp = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                intrinsic,
                extrinsic,
            )
            pcd += pcd_temp
            # TODO: move into config
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
        # possible additional processing steps
        # pcd = pcd.remove_non_finite_points()
        # pcd = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        # pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # pcd = pcd.remove_duplicated_points()
        # pcd = pcd.remove_duplicated_triangles()
        return pcd

    def visualize(self) -> None:
        '''
        Visualizes the reconstructed scene using native Open3d viewer.
        '''
        if not hasattr(self, 'pcd'):
            raise AttributeError("Point cloud is not reconstructed yet."
                                 " Please run reconstruct() first.")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.pcd)
        self._visualize_cameras(vis)
        vis.run()
        vis.destroy_window()

    def _visualize_cameras(self, vis) -> None:
        '''
        Visualizes cameras in the Open3d viewer
        using LineSet.create_camera_visualization.
        '''
        for camera in self.cameras.values():
            intrinsic = camera.get_o3d_intrinsic()
            extrinsic = camera.extrinsic
            line_set = o3d.geometry.LineSet.create_camera_visualization(
                intrinsic, extrinsic, scale=0.1)
            vis.add_geometry(line_set)

    def save(self, save_path: Path) -> None:
        '''
        Saves the reconstructed scene into COLMAP format.
        Copies images to 'images' folder of reconstruction.
        '''
        if not hasattr(self, 'pcd'):
            raise AttributeError("Point cloud is not reconstructed yet."
                                 " Please run reconstruct() first.")
        sparse_path = save_path / 'sparse' / '0'
        if not sparse_path.exists():
            sparse_path.mkdir(parents=True,  exist_ok=True)
        copy_images(self.images, save_path / 'images')
        write_cameras_binary(self.cameras, sparse_path / 'cameras.bin')
        write_images_binary(self.cameras, self.images, sparse_path / 'images.bin')
        write_points3D_binary(self.pcd, sparse_path / 'points3D.bin')
        logger.info(f"Saved reconstructed scene to {save_path} (BIN)")

    def save_txt(self, save_path: Path):
        '''
        Saves the reconstructed scene into COLMAP TXT format.
        Copies images to 'images' folder of reconstruction.
        '''
        if not hasattr(self, 'pcd'):
            raise AttributeError("Point cloud is not reconstructed yet."
                                 " Please run reconstruct() first.")
        sparse_path = save_path / 'sparse' / '0'
        if not sparse_path.exists():
            sparse_path.mkdir(parents=True, exist_ok=True)
        copy_images(self.images, save_path / 'images')
        write_cameras_text(self.cameras, sparse_path / 'cameras.txt')
        write_images_text(self.cameras, self.images, sparse_path / 'images.txt')
        write_points3D_text(self.pcd, sparse_path / 'points3D.txt')
        logger.info(f"Saved reconstructed scene to {save_path} (TXT)")
