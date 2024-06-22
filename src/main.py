import datetime
import time
import traceback

from runners import video_trainer
from runners import phototourism_trainer
from runners import static_trainer
from utils.create_rendering import render_to_path, decompose_space_time
from utils.parse_args import parse_optfloat

import argparse
import importlib.util
import logging
import os
import pprint
import random
import sys
from pathlib import Path
from typing import List, Dict, Any
import tempfile

import numpy as np
import torch
import torch.utils.data

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_freer_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fname = os.path.join(tmpdir, "tmp")
        os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >"{tmp_fname}"')
        if os.path.isfile(tmp_fname):
            memory_available = [int(x.split()[2]) for x in open(tmp_fname, 'r').readlines()]
            if len(memory_available) > 0:
                return np.argmax(memory_available)
    return None  # The grep doesn't work with all GPUs. If it fails we ignore it.

gpu = get_freer_gpu()
if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"CUDA_VISIBLE_DEVICES set to {gpu}")
else:
    print(f"Did not set GPU.")


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers,
                        force=True)


def load_data(model_type: str, data_downsample, data_dirs, validate_only: bool, validate_train_only: bool, render_only: bool, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)

    if model_type == "video":
        return video_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only, validate_train_only=validate_train_only,
            render_only=render_only, **kwargs)
    elif model_type == "phototourism":
        return phototourism_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only, validate_train_only=validate_train_only,
            render_only=render_only, **kwargs
        )
    else:
        return static_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only, validate_train_only=validate_train_only,
            render_only=render_only, **kwargs)


def init_trainer(model_type: str, **kwargs):
    if model_type == "video":
        from runners import video_trainer
        return video_trainer.VideoTrainer(**kwargs)
    elif model_type == "phototourism":
        from runners import phototourism_trainer
        return phototourism_trainer.PhototourismTrainer(**kwargs)
    else:
        from runners import static_trainer
        return static_trainer.StaticTrainer(**kwargs)


def save_config(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
        for key in config.keys():
            f.write("%s\t%s\n" % (key, config[key]))


def init_seeds(seed: int = 0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def main():
    setup_logging()

    p = argparse.ArgumentParser(description="")

    p.add_argument('--render-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--validate-train-only', action='store_true')
    p.add_argument('--spacetime-only', action='store_true')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('override', nargs=argparse.REMAINDER)

    args = p.parse_args()

    # Set random seed
    init_seeds(args.seed)

    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config
    # Process overrides from argparse into config
    # overrides can be passed from the command line as key=value pairs. E.g.
    # python src/main.py --config-path src/config/cfg.py max_ts_frames=200
    # note that all values are strings, so code should assume incorrect data-types for anything
    # that's derived from config - and should not a string.
    overrides: List[str] = args.override
    overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}
    config.update(overrides_dict)
    if "keyframes" in config:
        model_type = "video"
    elif "appearance_embedding_dim" in config:
        model_type = "phototourism"
    else:
        model_type = "static"
    validate_only = args.validate_only
    validate_train_only = args.validate_train_only
    render_only = args.render_only
    spacetime_only = args.spacetime_only
    if validate_only and render_only:
        raise ValueError("render_only and validate_only are mutually exclusive.")
    if render_only and spacetime_only:
        raise ValueError("render_only and spacetime_only are mutually exclusive.")
    if validate_only and spacetime_only:
        raise ValueError("validate_only and spacetime_only are mutually exclusive.")
    if validate_train_only and render_only:
        raise ValueError("validate_train_only and render_only are mutually exclusive.")
    if validate_train_only and validate_only:
        raise ValueError("validate_train_only and validate_only are mutually exclusive.")
    if validate_train_only and spacetime_only:
        raise ValueError("validate_train_only and spacetime_only are mutually exclusive.")

    pprint.pprint(config)
    if validate_only or validate_train_only or render_only:
        if args.log_dir is None and config['logdir'] is not None:
            args.log_dir = (Path(config['logdir']) / config['expname']).as_posix()
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
    else:
        save_config(config)

    data = load_data(model_type, validate_only=validate_only, validate_train_only=validate_train_only, render_only=render_only or spacetime_only, **config)
    config.update(data)
    trainer = init_trainer(model_type, **config)
    if args.log_dir is not None:
        checkpoint_path = os.path.join(args.log_dir, 'saved_models', "model.pth")
        training_needed = not (validate_only or validate_train_only or render_only or spacetime_only)
        trainer.load_model(torch.load(checkpoint_path), training_needed=training_needed)

    if validate_only or validate_train_only:
        trainer.validate()
    elif render_only:
        render_to_path(trainer, extra_name="")
    elif spacetime_only:
        decompose_space_time(trainer, extra_name="")
    else:
        trainer.train()
    return args.config_path


if __name__ == "__main__":
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        config_path = main()
        run_result = f'Program completed successfully! {config_path}'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
