# Shree KRISHNAya Namaha
# Undistorts the videos using the distortion coefficients and the method provided by InterDigital
# Author: Nagabhushan S N
# Last Modified: 13/01/24

import time
import datetime
import traceback
import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

from database_utils.interdigital.libraries.pose_warping.Warper import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_filenum = int(this_filename[-2:])

NUM_VIEWS = 16


class VideoUndistorer:
    def __init__(self):
        self.warper = Warper()
        return

    def undistort_videos(self, unzipped_dirpath: Path, database_dirpath: Path):
        for scene_dirpath in sorted(database_dirpath.iterdir()):
            scene_name = scene_dirpath.name

            # if scene_name not in ['Birthday', 'Painter', 'Remy', 'Theater', 'Train']:
            #     continue

            camera_params_path = unzipped_dirpath / f'{scene_name}/cameras_parameters.txt'
            camera_params_data = self.read_camera_params(camera_params_path)
            intrinsics_path = scene_dirpath / 'CameraIntrinsics.csv'
            intrinsics = self.read_intrinsics(intrinsics_path)
            for view_num in range(NUM_VIEWS):
                undistorted_video_path = scene_dirpath / f'rgb_undistorted{this_filenum:02}/{view_num:04}.mp4'
                if undistorted_video_path.exists():
                    continue

                d1, d2 = camera_params_data['d1'][view_num], camera_params_data['d2'][view_num]
                intrinsic = intrinsics[view_num]
                undistorted_frames = []
                frame_paths = sorted(unzipped_dirpath.glob(f'{scene_name}/{scene_name}_*_{view_num:02}.png'))
                for frame_path in tqdm(frame_paths, desc=f'{scene_name}_video{view_num:04}'):
                    frame = self.read_image(frame_path)
                    undistorted_frame = self.undistort_frame(frame, d1, d2, intrinsic)
                    undistorted_frames.append(undistorted_frame)
                undistorted_video = numpy.stack(undistorted_frames, axis=0)
                undistorted_video_path.parent.mkdir(parents=True, exist_ok=True)
                skvideo.io.vwrite(undistorted_video_path.as_posix(), undistorted_video,
                                  inputdict={'-r': str(30)},
                                  outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p'})
        return

    def undistort_frame(self, frame: numpy.ndarray, d1: float, d2: float, intrinsic: numpy.ndarray):
        h, w = frame.shape[:2]
        intrinsic = intrinsic[None, None]
        intrinsic_inv = numpy.linalg.inv(intrinsic)  # (1, 1, 3, 3)

        x, y = numpy.meshgrid(numpy.arange(w), numpy.arange(h))
        ones = numpy.ones_like(x)
        undist_pos_homo = numpy.stack([x, y, ones], axis=-1)[:, :, :, None]  # (h, w, 3, 1)
        undist_pos_homo_unnormalized = numpy.matmul(intrinsic_inv, undist_pos_homo)  # (h, w, 3, 1)
        dist_pos_unnormalized = self.dist_warp(undist_pos_homo_unnormalized[:, :, :2, 0], d1, d2)  # (h, w, 2)
        dist_pos_homo_unnormalized = numpy.concatenate([dist_pos_unnormalized, ones[:, :, None]], axis=2)[:, :, :, None]  # (h, w, 3, 1)
        dist_pos_homo_normalized = numpy.matmul(intrinsic, dist_pos_homo_unnormalized)  # (h, w, 3, 1)
        dist_pos = dist_pos_homo_normalized[:, :, :2, 0]
        grid = self.warper.create_grid(h, w)
        flow = dist_pos - grid

        undistorted_frame, mask = self.warper.bilinear_interpolation(frame.astype(numpy.float32), None, flow, None, is_image=True)
        return undistorted_frame

    @staticmethod
    def dist_warp(undist_pos: numpy.ndarray, d1: float, d2: float):
        """
        undist_pos: (h, w, 2)
        """
        r = numpy.linalg.norm(undist_pos, ord=2, axis=2, keepdims=True)
        multiplier = 1 + d1 * numpy.power(r, 2) + d2 * numpy.power(r, 4)
        dist_pos = multiplier * undist_pos
        return dist_pos

    @staticmethod
    def read_image(image_path: Path):
        image = skimage.io.imread(image_path.as_posix())
        return image

    @staticmethod
    def read_camera_params(camera_params_path: Path):
        camera_params_data = pandas.read_csv(camera_params_path, sep='\s+', header=None, skiprows=1)
        camera_params_data.columns = ['f', 'cu', 'cv', 'ar', 'sk', 'd1', 'd2', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']
        return camera_params_data

    @staticmethod
    def read_intrinsics(intrinsics_path: Path):
        intrinsics = numpy.loadtxt(intrinsics_path, delimiter=',').reshape((-1, 3, 3))
        return intrinsics


def demo1():
    root_dirpath = Path('../../')
    unzipped_dirpath = root_dirpath / 'data/all/database_data_unzipped'
    database_data_path = root_dirpath / 'data/all/database_data'
    undistorter = VideoUndistorer()
    undistorter.undistort_videos(unzipped_dirpath, database_data_path)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

    from snb_utils import Telegrammer

    time.sleep(5)
    message_content = f'R21/ID/{this_filename} has finished.\n' + run_result
    Telegrammer.send_message(message_content, chat_names=['Nagabhushan'])
