# Shree KRISHNAya Namaha
# Visualizes valid flow by warping the frame
# Author: Nagabhushan S N
# Last Modified: 25/12/23

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

from pose_warping.Warper import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class ValidFlowVisualizer:
    def __init__(self, database_dirpath: str, flow_dirname: str, mask_dirname: str):
        self.database_dirpath = database_dirpath
        self.flow_dirname = flow_dirname
        self.mask_dirname = mask_dirname

        self.warper = Warper()
        return

    def visualize_valid_flow(self, scene_name: str, video1_num: int, frame1_num: int, video2_num: int, frame2_num: int):
        root_dirpath = Path('../../')
        project_dirpath = root_dirpath / '../../../../'
        database_dirpath = project_dirpath / 'databases' / self.database_dirpath
        tmp_dirpath = root_dirpath / 'tmp'

        frame1_path = tmp_dirpath / 'rgb_frames' / f'{scene_name}/{video1_num:04}/{frame1_num:04}.png'
        frame2_path = tmp_dirpath / 'rgb_frames' / f'{scene_name}/{video2_num:04}/{frame2_num:04}.png'
        flow12_path = database_dirpath / f'all/estimated_flows/{self.flow_dirname}/{scene_name}/estimated_flows/{video1_num:04}_{frame1_num:04}__{video2_num:04}_{frame2_num:04}.npz'
        mask_path = database_dirpath / f'all/estimated_flow_masks/{self.mask_dirname}/{scene_name}/valid_masks/{video1_num:04}_{frame1_num:04}__{video2_num:04}_{frame2_num:04}.png'

        frame1 = self.read_image(frame1_path)
        frame2 = self.read_image(frame2_path)
        flow12 = self.read_flow(flow12_path)
        mask = self.read_mask(mask_path)

        warped_frame1, mask1 = self.warper.bilinear_interpolation(frame2, None, flow12, mask, is_image=True)
        mask1_3d = mask1[:, :, None]
        warped_frame1 = mask1_3d * warped_frame1 + (~mask1_3d) * frame1

        skimage.io.imsave(f'{scene_name}__{video1_num:04}_{frame1_num:04}.png', frame1)
        skimage.io.imsave(f'{scene_name}__{video2_num:04}_{frame2_num:04}.png', frame2)
        skimage.io.imsave(f'{scene_name}__{video1_num:04}_{frame1_num:04}__{video2_num:04}_{frame2_num:04}.png', warped_frame1)
        return

    @staticmethod
    def read_image(path: Path):
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def read_mask(path: Path):
        mask = skimage.io.imread(path.as_posix()) == 255
        return mask

    @staticmethod
    def read_flow(path: Path):
        match path.suffix:
            case '.npy':
                flow = numpy.load(path.as_posix())
            case '.npz':
                flow = numpy.load(path.as_posix())['arr_0']
            case _:
                raise ValueError(f'Invalid flow file extension: {path.suffix}')
        return flow


def demo1():
    database_dirpath = 'N3DV/data'
    flow_dirname = 'FEL003_FE04'
    mask_dirname = 'FEL003_FV04'

    scene_name = 'cut_roasted_beef'
    video1_num = 5
    frame1_num = 140
    video2_num = 5
    frame2_num = 150

    flow_visualizer = ValidFlowVisualizer(database_dirpath, flow_dirname, mask_dirname)
    flow_visualizer.visualize_valid_flow(scene_name, video1_num, frame1_num, video2_num, frame2_num)
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
