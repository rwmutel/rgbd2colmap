from ..camera import Camera
from ..images import Image
from ..depths import Depth
from typing import Dict


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
        self.cameras = cameras
        self.rgbds = self.combine_images_and_depths(images, depths)

    def combine_images_and_depths(
        self,
        images: Dict[str | int, Image],
        depths: Dict[str | int, Depth],
    ) -> Dict[str | int, Image]:
        '''
        Combines images and depths into a single dictionary
        using the same keys for both.
        '''
        pass

    def reconstruct(self):
        '''
        Reconstructs scene point cloud from cameras, images and depths.
        Optionally includes pre- and postprocessing steps.
        '''
        # self.preprocess()
        # self.reconstruct_scene()
        # self.postprocess()
        pass

    def reconstruct_scene(self):
        '''
        Reconstructs the scene by unprojecting depth maps from given cameras
        '''
        pass

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