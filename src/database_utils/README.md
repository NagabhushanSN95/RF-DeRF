# Extracting Databases

## N3DV
1. Download the dataset files from [here](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) and place the downloaded files at `data/databases/N3DV/data/raw/downloaded_data/`.

2. Unzip each of the zip files and place the unzipped files at `data/databases/N3DV/data/raw/unzipped_data`. The file structure should look like below
    ```shell
    data
    |--databases
    |  |--N3DV
    |  |  |--data
    |  |  |  |--raw
    |  |  |  |  |--downloaded_data
    |  |  |  |  |--unzipped_data
    |  |  |  |  |  |--coffee_martini
    |  |  |  |  |  |  |--cam00.mp4
    |  |  |  |  |  |  |--cam01.mp4
    |  |  |  |  |  |  ...
    |  |  |  |  |  |  |--poses_bounds.npy
    |  |  |  |  |  |--cook_spinach
    |  |  |  |  |  ...
    ```

3. Run the data extractor file:
   ```shell
   cd src/database_utils/n3dv/data_organizers/
   python DataExtractor01.py
   cd ..
   ```
   
4. Following [K-Planes](https://github.com/sarafridov/K-Planes), we recommend running the experiments at downsampled 2 resolution. To avoid downsampling during every run, downsample the videos upfront
    ```shell
    cd data_processors
    python VideoDownsampler.py
    cd ..
    ```

5. train/test configs are already provided in the repository. In case you want to create them again: 
   ```shell
   cd train_test_creators/
   python TrainTestCreator01_UniformSparseSampling.py
   cd ..
   ```

6. Return to root directory
   ```shell
   cd ../../../
   ```

## Inter Digital
1. Download the dataset files from [here](https://www.interdigital.com/data_sets/light-field-dataset) and place the downloaded files at `data/databases/InterDigital/data/raw/downloaded_data/`.

2. Unzip each of the zip files and place the unzipped files at `data/databases/InterDigital/data/raw/unzipped_data`. The file structure should look like below
   ```shell
   data
   |--databases
   |  |--InterDigital
   |  |  |--data
   |  |  |  |--raw
   |  |  |  |  |--downloaded_data
   |  |  |  |  |--unzipped_data
   |  |  |  |  |  |--Birthday
   |  |  |  |  |  |  |--Birthday_00000_00.png
   |  |  |  |  |  |  |--Birthday_00000_01.png
   |  |  |  |  |  |  ...
   |  |  |  |  |  |  |--camera_parameters.txt
   |  |  |  |  |  |--Painter
   |  |  |  |  |  ...
   ```

3. Run the data extractor file:
   ```shell
   cd src/database_utils/interdigital/data_organizers/
   python DataExtractor01.py
   cd ..
   ```
   
4. Following [K-Planes](https://github.com/sarafridov/K-Planes), we recommend running the experiments at downsampled 2 resolution. To avoid downsampling during every run, downsample the videos upfront
   ```shell
   cd data_processors
   python VideoDownsampler.py
   cd ..
   ```
   
5. If you have downloaded the undistorted images, you can skip this step. If you have downloaded the raw/distorted images, undistort the images using the distortion parameters given in the dataset.
    ```shell
   cd data_processors
   python VideoUndistorter01.py
   cd ..
   ```

6. train/test configs are already provided in the repository. In case you want to create them again: 
   ```shell
   cd train_test_creators/
   python TrainTestCreator01_UniformSparseSampling.py
   cd ..
   ```

7. Return to root directory
   ```shell
   cd ../../../
   ```

## Custom Databases
We use the Open CV convention: `(x, -y, -z)` world-to-camera format to store the camera poses. 
The camera intrinsics and extrinsics are stored in the `csv` format after flattening them, i.e., if a scene contains 50 videos, intrinsics and extrinsics are stores as csv files with 50 rows each and 9 & 16 columns respectively.
The directory tree in the following shows an example.
Please refer to the [video_datasets.py](../datasets/video_datasets.py) for more details. 
Organize your custom dataset in accordance with the data-loader or write a new data-loader file to load the data directly from your custom database format.

Example directory tree:
```shell
<DATABASE_NAME>
 |--data
    |--all
    |  |--database_data
    |     |--scene0001
    |     |  |--rgb
    |     |  |  |--0000.mp4
    |     |  |  |--0001.mp4
    |     |  |  |-- ...
    |     |  |--CameraExtrinsics.csv
    |     |  |--CameraIntrinsics.csv
    |     |--scene0002
    |     | ...
    |--train_test_sets
```

Our code also requires a config file specifying the train/validation/test images. Please look into [train-test-creators](n3dv/train_test_creators/TrainTestCreator01_UniformSparseSampling.py) and replicate a similar file for your custom dataset.
