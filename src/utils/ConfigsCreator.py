# Shree KRISHNAya Namaha
# Takes Configs.py and creates new configs file in scene_name for every scene. This can then be used by bash file to call individual train and test commands.
# Author: Nagabhushan S N
# Last Modified: 30/10/2023

import argparse
import importlib
import importlib.util
import json
import shutil
from collections import OrderedDict
from typing import Dict, Any

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_configs(configs_path: Path):
    spec = importlib.util.spec_from_file_location(configs_path.stem, configs_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    configs: Dict[str, Any] = cfg.config
    return configs


def save_configs(scene_configs_path: Path, scene_configs: dict):
    dumped_configs = json.dumps(scene_configs, indent=4)
    dumped_configs = dumped_configs.replace('true', 'True').replace('false', 'False').replace('"', "'")
    dumped_configs = dumped_configs.replace('_True_', '_true_')
    with open(scene_configs_path.as_posix(), 'w') as scene_configs_file:
        scene_configs_file.write('config = ')
        scene_configs_file.writelines(dumped_configs)
        # json.dump(scene_configs, scene_configs_file, indent=4)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs-path', type=str)
    parser.add_argument('--scene-names', type=str, nargs='*')

    args = parser.parse_args()
    configs_path = Path(args.configs_path)
    print(configs_path)
    train_dirpath = configs_path.parent

    configs = read_configs(configs_path)
    data_dirpath = Path(configs['data_dirpath'])

    for scene_name in args.scene_names:
        scene_dirpath = train_dirpath / f'{scene_name}'
        scene_dirpath.mkdir(parents=True, exist_ok=False)

        scene_configs = OrderedDict(configs.copy())
        del scene_configs['data_dirpath']
        scene_configs['expname'] = scene_name
        scene_configs['logdir'] = train_dirpath.as_posix()
        scene_configs['data_dirs'] = [(data_dirpath / f'database_data/{scene_name}').as_posix()]
        scene_configs['flow_dirpath'] = (data_dirpath / f'estimated_flows').as_posix()
        scene_configs['depth_dirpath'] = (data_dirpath / f'estimated_depths').as_posix()
        scene_configs.move_to_end('depth_dirpath', last=False)
        scene_configs.move_to_end('flow_dirpath', last=False)
        scene_configs.move_to_end('data_dirs', last=False)
        scene_configs.move_to_end('logdir', last=False)
        scene_configs.move_to_end('expname', last=False)
        scene_configs_path = scene_dirpath / 'dynerf.py'
        save_configs(scene_configs_path, scene_configs)

    pychache_dirpath = train_dirpath / '__pycache__'
    if pychache_dirpath.exists():
        shutil.rmtree(pychache_dirpath)
    return


if __name__ == '__main__':
    main()
