# RF-DeRF
Official code release accompanying the SIGGRAPH 2024 paper "Factorized Motion Fields for Fast Sparse Input Dynamic View Synthesis"

* [Project Page](https://nagabhushansn95.github.io/publications/2024/RF-DeRF.html)
* [Published Data (OneDrive)](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/nagabhushans_iisc_ac_in/EkUVNYjCZq9Mh_fZa-64Co0B2MyXpswE3BjZFuf2eBhlYA?e=dWqELA)

## Setup

### Python Environment
Environment details are available in `EnvironmentData/RF_DeRF.yml`. The environment can be created using conda
```shell
conda env create -f RF_DeRF.yml
```

### Add the source directory to PYTHONPATH
```shell
export PYTHONPATH=<ABSOLUTE_PATH_TO_RFDERF_DIR>/src:$PYTHONPATH
```

### Set-up Databases
Please follow the instructions in [database_utils/README.md](src/database_utils/README.md) file to set up various databases. Instructions for custom databases are also included here.

### Generate Priors
Please follow the instructions in [prior_generators/README.md](src/prior_generators/README.md) file to generate sparse flow and dense flow priors.

## Training and Inference
To train a model and run the inference, execute the following commands
```shell
cd src/
bash TrainerTester01_N3DV.sh path/to/Configs.py
bash TrainerTester02_ID.sh path/to/Configs.py
cd ..
```
For example
```shell
cd src/
bash TrainerTester01_N3DV.sh ../runs/training/train0006/Configs.py
cd ..
```

Similar to K-Planes, it is recommended to first run for a single iteration at 4x downsampling to pre-compute and store the ray importance weights, and then run as usual at 2x downsampling. 

### Inference with Pre-trained Models
The train configs are also provided in `runs/training/train****` folders for each of the scenes. Please download the trained models from `runs/training` directory in the published data (link available at the top) and place them in the appropriate folders. Then execute the following commands
```shell
cd src
python main.py --config-path path/to/Configs.py --validation-only
python main.py --config-path path/to/Configs.py --render-only
cd ..
```

### Evaluation
Evaluation of the rendered images will be automatically done after rendering the images. 

To compute depth based metrics, ground truth depth maps are needed. We obtain (pseudo) ground truth depth maps by training the vanilla K-Planes with dense input views. Download these depth maps from `data` directory in the published data (link available at the top) and place them in the appropriate folders. Update the `pred_train_dirpath` in [AllMetrics01_N3DV.py](src/qa/00_Common/src/AllMetrics01_N3DV.py) or [AllMetrics02_ID.py](src/qa/00_Common/src/AllMetrics02_ID.py) file appropriately.
```shell
cd src/qa
python AllMetrics01_N3DV.py
python AllMetrics02_ID.py
cd ../..
```

## License
MIT License

Copyright (c) 2024 Nagabhushan Somraj, Kapil Choudhary, Sai Harsha Mupparaju, Rajiv Soundararajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Citation
If you use this code for your research, please cite our paper

```bibtex
@inproceedings{somraj2024rfderf,
    title = {Factorized Motion Fields for Fast Sparse Input Dynamic View Synthesis},
    author = {Somraj, Nagabhushan and Choudhary, Kapil and Mupparaju, Harsha and Soundararajan, Rajiv},
    booktitle = {SIGGRAPH},
    month = {July},
    year = {2024},
    doi = {10.1145/3641519.3657498},
}
```
If you use outputs/results of RF-DeRF model in your publication, please specify the version as well. The current version is 1.0.

## Acknowledgements
Our code is built on top of [K-Planes](https://github.com/sarafridov/K-Planes) codebase.


For any queries or bugs regarding RF-DeRF, please raise an issue.
