import logging
from pathlib import Path
from typing import Dict, Tuple
import copy

import cv2
import open3d as o3d
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from ..camera import Camera
from ..depths import Depth
from ..images import Image
from ..utils.colmap_io import (copy_images, write_cameras_binary,
                               write_cameras_text, write_images_binary,
                               write_images_text, write_points3D_binary,
                               write_points3D_text)

logger = logging.getLogger(__name__)


class RGBDReconstructionParams:
    '''
    Parameters for RGBD reconstruction parsed from OmegaConf config.
    '''
    def __init__(self, cfg: DictConfig):
        self.target_image_size = cfg.target_image_size if 'target_image_size' in cfg else None
        self.voxel_size = cfg.voxel_size if 'voxel_size' in cfg else 0.05
        self.max_depth = cfg.max_depth if 'max_depth' in cfg else 1000.0
        self.icp_registration = 'icp_registration' in cfg and cfg.icp_registration
        if self.icp_registration:
            self.relative_fitness = cfg.relative_fitness if 'relative_fitness' in cfg else 1e-6
            self.relative_rmse = cfg.relative_rmse if 'relative_rmse' in cfg else 1e-6
            self.max_iterations = cfg.max_iterations if 'max_iterations' in cfg else 30


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
        parameters: RGBDReconstructionParams | DictConfig,
    ):
        if isinstance(parameters, DictConfig):
            parameters = RGBDReconstructionParams(parameters)
        self.parameters = parameters
        if parameters.target_image_size is not None:
            self.target_width, self.target_height = parameters.target_image_size
        else:
            self.target_width, self.target_height = self._get_image_shape(images)
        self.cameras = self._rescale_intrinsics(cameras)
        self.images = self._rescale_images(images)
        self.rgbds = self._combine_images_and_depths(images, depths)

    def _rescale_intrinsics(
        self,
        cameras: Dict[str | int, Camera]
    ) -> Dict[str | int, Camera]:
        '''
        Rescales camera intrinsics to match the target image size.
        Assumes all cameras have the same intrinsic matrix.
        '''
        original_width, original_height = self._get_image_shape(self.images)
        for camera in cameras.values():
            if camera.width != self.target_width \
               or camera.height != self.target_height:
                camera.width = self.target_width
                camera.height = self.target_height
                camera.intrinsics[0, :] *= self.target_width / original_width
                camera.intrinsics[1, :] *= self.target_height / original_height
        return cameras

    def _get_image_shape(
        self,
        images: Dict[str | int, Image]
    ) -> Tuple[int, int]:
        '''
        Returns width and height of the first image in the dictionary.
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
                    depth_trunc=self.parameters.max_depth,
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

    def _rescale_images(
        self,
        images: Dict[str | int, Image]
    ) -> Dict[str | int, Image]:
        '''
        Rescales images to match the target image size.
        Assumes all images have the same shape.
        '''
        for key, image in images.items():
            image_height, image_width, _ = image.image_np.shape
            if image_width != self.target_width \
               or image_height != self.target_height:
                image.image_np = cv2.resize(
                    image.image_np,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_CUBIC,
                )
        return images

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
            pcd = pcd.voxel_down_sample(
                voxel_size=self.parameters.voxel_size)
            pcd_temp = pcd_temp.voxel_down_sample(
                voxel_size=self.parameters.voxel_size)
            if self.parameters.icp_registration and len(pcd.points) > 0:
                extrinsic = self._get_icp_transform(pcd, pcd_temp)
                pcd_temp.transform(extrinsic)

                # print(extrinsic)
                # vis = o3d.visualization.Visualizer()
                # vis.create_window()
                # vis.add_geometry(copy.deepcopy(pcd).paint_uniform_color([0, 1, 0]))
                # vis.add_geometry(copy.deepcopy(pcd_temp).paint_uniform_color([1, 0, 0]))
                # vis.add_geometry(copy.deepcopy(pcd_temp.transform(extrinsic).paint_uniform_color([0, 0, 1])))
                # vis.run()
                # vis.destroy_window()
            pcd += pcd_temp

        # TODO possible additional processing steps
        # pcd = pcd.remove_non_finite_points()
        # pcd = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        # pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # pcd = pcd.remove_duplicated_points()
        # pcd = pcd.remove_duplicated_triangles()
        return pcd

    def _get_icp_transform(
        self,
        global_pcd: o3d.geometry.PointCloud,
        registring_pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        '''
        Registers point cloud using ICP colored registration.
        Returns the adjusting transformation for registring_pcd
        to align with global_pcd.
        https://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
        https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf
        '''
        for pcd in [global_pcd, registring_pcd]:
            if not pcd.has_normals():
                pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.parameters.voxel_size * 2, max_nn=30))
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            registring_pcd, global_pcd, self.parameters.voxel_size, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.parameters.relative_rmse,
                relative_rmse=self.parameters.relative_rmse,
                max_iteration=self.parameters.max_iterations))
        logger.info(f"ICP registration result: {result_icp}")
        return result_icp.transformation

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
