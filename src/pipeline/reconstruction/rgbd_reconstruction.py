from ..camera import Camera
from ..images import Image
from ..depths import Depth
from typing import Dict, Tuple
import open3d as o3d
import numpy as np
import logging
import cv2
from tqdm import tqdm

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
        self.target_width, self.target_height = self._get_image_shape(images)  # TODO: move into config, implement image rescaling as well as depth map
        self.cameras = self._rescale_intrinsics(cameras)
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
        return first_image.shape[1], first_image.shape[0]

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
                depth_o3d = self._rescale_image(depths[key])
                image_o3d = o3d.geometry.Image(images[key])
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

    def _rescale_image(self, image: Depth) -> o3d.geometry.Image:
        '''
        Rescales depth image to match the image shape.
        Assumes camera intrisics are given for the RGB image
        '''
        image = cv2.resize(
            image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC,
        )
        image = o3d.geometry.Image(image)
        return image

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
            pcd = pcd.voxel_down_sample(voxel_size=0.05) # TODO: move into config
        # possible additional processing steps
        # pcd = pcd.remove_non_finite_points()
        # pcd = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        # pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # pcd = pcd.remove_duplicated_points()
        # pcd = pcd.remove_duplicated_triangles()
        return pcd

    def visualize(self):
        '''
        Visualizes the reconstructed scene using native Open3d viewer.
        '''
        pass

    def save(self):
        '''
        Saves the reconstructed scene into COLMAP format.
        '''
        pass