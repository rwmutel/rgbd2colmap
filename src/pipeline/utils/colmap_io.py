# This code relies heavily on original code for read/write operations
# from COLMAP (https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py),
# which is licensed under the following terms:

# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.

import collections
import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d

from ..camera import Camera
from ..images import Image

CAMERA_MODEL_NAME = "PINHOLE"
CAMERA_MODEL_ID = 1


# def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
#     """pack and write to a binary file.
#     :param fid:
#     :param data: data to send, if multiple elements are sent at the same time,
#     they should be encapsuled either in a list or a tuple
#     :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
#     should be the same length as the data list or tuple
#     :param endian_character: Any of {@, =, <, >, !}
#     """
#     if isinstance(data, (list, tuple)):
#         bytes = struct.pack(endian_character + format_char_sequence, *data)
#     else:
#         bytes = struct.pack(endian_character + format_char_sequence, data)
#     fid.write(bytes)


# def write_cameras_binary(cameras: Dict[str | int, Camera], path: Path | str):
#     """
#     see: src/colmap/scene/reconstruction.cc
#         void Reconstruction::WriteCamerasBinary(const std::string& path)
#         void Reconstruction::ReadCamerasBinary(const std::string& path)
#     """
#     with open(path, "wb") as fid:
#         write_next_bytes(fid, len(cameras), "Q")
#         for cam_id, cam in cameras.items():
#             model_id = CAMERA_MODEL_ID
#             camera_properties = [cam_id, model_id, cam.width, cam.height]
#             write_next_bytes(fid, camera_properties, "iiQQ")
#             for p in cam.params:
#                 write_next_bytes(fid, float(p), "d")
#     return cameras


def write_cameras_text(cameras: Dict[int | str, Camera], path: Path | str):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for cam_id, cam in cameras.items():
            to_write = [cam_id, CAMERA_MODEL_NAME, cam.width, cam.height,
                        cam.intrinsic[0, 0], cam.intrinsic[1, 1],
                        cam.intrinsic[0, 2], cam.intrinsic[1, 2]]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


# def write_images_binary(images: Dict[int | str, Image], path: Path | str):
#     """
#     see: src/colmap/scene/reconstruction.cc
#         void Reconstruction::ReadImagesBinary(const std::string& path)
#         void Reconstruction::WriteImagesBinary(const std::string& path)
#     """
#     with open(path, "wb") as fid:
#         write_next_bytes(fid, len(images), "Q")
#         for img_id, img in images.items():
#             write_next_bytes(fid, img_id, "i")
#             write_next_bytes(fid, img.qvec.tolist(), "dddd")
#             write_next_bytes(fid, img.tvec.tolist(), "ddd")
#             write_next_bytes(fid, img.camera_id, "i")
#             for char in img.name:
#                 write_next_bytes(fid, char.encode("utf-8"), "c")
#             write_next_bytes(fid, b"\x00", "c")
#             write_next_bytes(fid, len(img.point3D_ids), "Q")
#             for xy, p3d_id in zip(img.xys, img.point3D_ids):
#                 write_next_bytes(fid, [*xy, p3d_id], "ddq")


def write_images_text(
    cameras: Dict[int | str, Camera],
    path: Path | str
):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    HEADER = (
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        "# Number of images: {}\n".format(len(cameras))
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for key in cameras.keys():
            cam = cameras[key]
            image_header = [
                key,
                *(cam.qvec()).tolist(),
                *(cam.tvec()).tolist(),
                key,
                f"{key:05d}.jpg",
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            # Single (point2D_x, point2D_y, pointID) to preserve colmap format
            points_strings.append(" ".join(map(str, [0.0, 0.0, -1])))
            fid.write(" ".join(points_strings) + "\n")


# def write_points3D_binary(points3D: o3d.geometry.PointCloud, path: Path | str):
#     """
#     see: src/colmap/scene/reconstruction.cc
#         void Reconstruction::ReadPoints3DBinary(const std::string& path)
#         void Reconstruction::WritePoints3DBinary(const std::string& path)
#     """
#     with open(path, "wb") as fid:
#         write_next_bytes(fid, len(points3D), "Q")
#         for pt_id, pt in enumerate(points3D):
#             write_next_bytes(fid, pt_id, "Q")
#             write_next_bytes(fid, pt[:3].tolist(), "ddd")
#             write_next_bytes(fid, pt[4:].tolist(), "BBB")
#             # reprojection error = 0 is kept to preserve the format
#             write_next_bytes(fid, 0, "d")
#             track_length = 1
#             write_next_bytes(fid, track_length, "Q")
#             # image_id = 0, point2D_id = 0 kept to preserve the format
#             write_next_bytes(fid, [0, 0], "ii")


def write_points3D_text(points3D: o3d.geometry.PointCloud, path: Path | str):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = np.hstack((np.asarray(points3D.points), np.asarray(points3D.colors)))
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as"
        " (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: {}\n".format(len(points3D))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for pt_id, pt in enumerate(points3D):
            # reprojection error = 0 is kept to preserve the format
            point_header = [pt_id, *pt[:3].tolist(), *(255*pt[3:6]).astype(np.int64).tolist(), 0]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            # (image_id, point2D_id) is kept to preserve the format
            track_strings.append(" ".join(map(str, [0, 0])))
            fid.write(" ".join(track_strings) + "\n")
