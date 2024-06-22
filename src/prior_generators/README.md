# Generation of Flow Priors

We employ cross-camera sparse flow and within-camera dense flow priors.

## Sparse Flow Priors
We use Colmap to generate sparse flow. Installation instructions can be found [here](https://colmap.github.io/install.html).
Run the following files to generate sparse flow priors for the respective datasets for all the three input configurations.
```shell
cd src/prior_generators/sparse_flow/data_generators
python FlowEstimator_N3DV01.py
python FlowEstimator_ID01.py
cd ../utils
python FlowAggregator.py
cd ../../../../
```

Running the above files creates a new directory `data/databases/<DATABASE_NAME>/data/all/estimated_flows`, which contains three sub-directories named `FEL001_FE03,FEL001_FE04,FEL001_FE05` corresponding to two, three and four input-view settings. Each of these directories will contain multiple sub-directories, one for every scene in the dataset. The following tree shows an exmaple.
```
data/databases/N3DV/data/all/estimated_flow
|--FEL001_FE03
|  |--coffee_martini
|  |  |--MatchedPixels.csv
|  |--cook_spinach
|  |  ...  
|--FEL001_FE04
|  ...
```

## Dense Flow Priors
We use RAFT to generate sparse flow. Installation instructions can be found [here](https://github.com/princeton-vl/RAFT).
Run the following files to generate sparse flow priors for the respective datasets for all the three input configurations.
```shell
cd src/prior_generators/sparse_flow/data_generators
python FlowEstimator_N3DV01.py
python FlowEstimator_ID01.py
cd ../../../../
```

Running the above files creates a new directory `data/databases/<DATABASE_NAME>/data/all/estimated_flows`, which contains three sub-directories named `FEL003_FE03,FEL003_FE04,FEL003_FE05` corresponding to two, three and four input-view settings. Each of these directories will contain multiple sub-directories, one for every scene in the dataset. The following tree shows an exmaple.
```
data/databases/N3DV/data/all/estimated_flow
|--FEL003_FE03
|  |--coffee_martini
|  |  |--estimated_flows
|  |  |  |--<view1:04>_0000__<view1>_0010.npz
|  |  |  |  ...
|  |  |  |--<view2:04>_0000__<view2>_0010.npz
|  |  |  |  ...
|  |--cook_spinach
|  |  ...  
|--FEL003_FE04
|  ...
```


## Acknowledgements
Parts of the code are borrowed from [DS-NeRF](https://github.com/dunbar12138/DSNeRF) and [RAFT](https://github.com/princeton-vl/RAFT) codebases.
