# ReFlow-DyRF
Official code release accompanying the SIGGRAPH 2024 paper "Factorized Motion Fields for Fast Sparse Input Dynamic View Synthesis"

* [Project Page](https://nagabhushansn95.github.io/publications/2024/ReFlow-DyRF.html)
* [Published Data (OneDrive)]()


## Python Environment Setup 

Environment details are available in EnvironmentData/ReFlow_DyRF.yml. The environment can be created using conda

```shell
conda env create -f ReFlow_DyRF.yml
```


## Set-up Databases

## Generate Priors

## Training

To train a model and run the inference, execute the following commands
```shell
PYTHONPATH='src'
cd src
python main.py --config-path path/to/config.py
python main.py --config-path path/to/config.py --render-only
```

Similar to K-Planes, it is recommended to first run for a single iteration at 4x downsampling to pre-compute and store the ray importance weights, and then run as usual at 2x downsampling. 

## Inference with Pre-trained Models

The train configs are also provided in runs/training/train**** folders for each of the scenes. Please download the trained models from runs/training directory in the published data (link available at the top) and place them in the appropriate folders. Then execute the following commands
```shell
PYTHONPATH='src'
cd src
python main.py --config-path path/to/config.py --validation-only
python main.py --config-path path/to/config.py --render-only
```


## License 
MIT License

Copyright (c) 2024 Nagabhushan Somraj, Kapil Choudhary, Harsha Mupparaju, Rajiv Soundararajan

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
@inproceedings{somraj2024ReFlowDyRF,
    title = {Factorized Motion Fields for Fast Sparse Input Dynamic View Synthesis},
    author = {Somraj, Nagabhushan and Choudhary, Kapil and Mupparaju, Harsha and Soundararajan, Rajiv},
    booktitle = {SIGGRAPH},
    month = {July},
    year = {2024},
    doi = {},
}
```
If you use outputs/results of ReFlow-DyRF model in your publication, please specify the version as well. The current version is 1.0.

## Acknowledgements
Our code is built on top of [K-Planes](https://github.com/sarafridov/K-Planes) codebase.


For any queries or bugs regarding ReFlow-DyRF, please raise an issue.
