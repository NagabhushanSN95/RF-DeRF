# Shree KRISHNAya Namaha
# Predicts sparse flow from the given images without using the camera extrinsics or intrinsics from the database.
# Extended from ImagesMatcher03.py and supports copying images given their paths instead of writing them to disk from
# RAM. Compatible with VideoFramesMatcher02.py.
# Author: Nagabhushan S N, Harsha Mupparaju
# Last Modified: 23/06/2024

import os
import shutil
import time
import datetime
import traceback
from itertools import combinations

import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path

from tqdm import tqdm
from matplotlib import pyplot
from typing import List, Union

from colmap_utils.read_write_model import read_images_binary, read_points3d_binary

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class ColmapTester:
    def __init__(self, tmp_dirpath: Path):
        self.tmp_dirpath = tmp_dirpath
        self.images_dirpath = self.tmp_dirpath / 'images'
        self.db_path = self.tmp_dirpath / 'database.db'
        self.sparse_dirpath = self.tmp_dirpath / 'sparse'
        self.sparse_dirpath1 = self.sparse_dirpath / '0'
        return
    
    def clean_tmp_dir(self):
        if self.tmp_dirpath.exists():
            shutil.rmtree(self.tmp_dirpath)
        self.tmp_dirpath.mkdir(parents=True)
        return 
    
    def save_tmp_data(self, images: List[Union[Path, numpy.ndarray]]):
        self.sparse_dirpath.mkdir(parents=True, exist_ok=True)
        self.images_dirpath.mkdir(parents=True, exist_ok=True)
        for frame_num, image in enumerate(images):
            tgt_image_path = self.images_dirpath / f'{frame_num:04}.png'
            if isinstance(image, Path):
                shutil.copy(image, tgt_image_path)
            elif isinstance(image, numpy.ndarray):
                self.save_image(tgt_image_path, image)
            else:
                raise RuntimeError
        return
    
    def run_colmap(self):
        cmd = f'colmap feature_extractor --database_path {self.db_path.as_posix()} --image_path {self.images_dirpath.as_posix()} --ImageReader.single_camera 1 --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true'
        print(cmd)
        os.system(cmd)

        cmd = f'colmap exhaustive_matcher --database_path {self.db_path.as_posix()}'
        print(cmd)
        os.system(cmd)

        cmd = f'colmap mapper --database_path {self.db_path.as_posix()} --image_path {self.images_dirpath.as_posix()} --output_path {self.sparse_dirpath.as_posix()} --Mapper.tri_ignore_two_view_tracks 0 --Mapper.num_threads 16 --Mapper.init_min_tri_angle 4 --Mapper.multiple_models 0 --Mapper.extract_colors 0'
        # cmd = f'colmap mapper --database_path {self.db_path.as_posix()} --image_path {self.images_dirpath.as_posix()} --export_path {self.sparse_dirpath1.as_posix()} --Mapper.tri_ignore_two_view_tracks 0 --Mapper.num_threads 16 --Mapper.init_min_tri_angle 4 --Mapper.multiple_models 0 --Mapper.extract_colors 0'
        print(cmd)
        os.system(cmd)

        pts_filepath = self.sparse_dirpath1 / 'points3D.bin'
        success = pts_filepath.exists()

        if success:
            cmd = f'colmap model_converter --input_path {self.sparse_dirpath1.as_posix()} --output_path {self.sparse_dirpath1.as_posix()} --output_type TXT'
            print(cmd)
            os.system(cmd)

        return success

    @staticmethod
    def read_image(path: Path) -> numpy.ndarray:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def save_image(path: Path, image: numpy.ndarray):
        skimage.io.imsave(path.as_posix(), image)
        return

    def get_sparse_matches(self):
        if not (self.sparse_dirpath1 / 'images.bin').exists():
            return None

        images = read_images_binary((self.sparse_dirpath1 / 'images.bin').as_posix())
        points = read_points3d_binary((self.sparse_dirpath1 / 'points3D.bin').as_posix())

        image1_id = next(filter(lambda x: x.name == '0000.png', images.values())).id
        image2_id = next(filter(lambda x: x.name == '0001.png', images.values())).id

        matched_points = []
        for point_3d_id in points:
            point_3d = points[point_3d_id]
            image_ids = point_3d.image_ids
            point_2d_idxs = point_3d.point2D_idxs

            image1_idx = numpy.where(image_ids == image1_id)[0].item()
            image2_idx = numpy.where(image_ids == image2_id)[0].item()
            pixel1_idx = point_2d_idxs[image1_idx]
            pixel2_idx = point_2d_idxs[image2_idx]

            pixel1_loc = images[image1_id].xys[pixel1_idx]
            pixel2_loc = images[image2_id].xys[pixel2_idx]
            error = point_3d.error.item()
            matched_points.append(pixel1_loc.tolist() + pixel2_loc.tolist() + [point_3d_id, error])
        matches_data = pandas.DataFrame(matched_points, columns=['x1', 'y1', 'x2', 'y2', '3D_id', 'error']).sort_values(by='3D_id', axis=0)
        return matches_data

    def estimate_sparse_flow(self, images: List[Union[Path, numpy.ndarray]], extrinsics: List[numpy.ndarray], intrinsics: List[numpy.ndarray]):
        self.clean_tmp_dir()
        self.save_tmp_data(images)
        success = self.run_colmap()
        if success:
            matches_data = self.get_sparse_matches()
        else:
            matches_data = None
        return matches_data
