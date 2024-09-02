# Source Code Description

Note: All paths mentioned here are relative to _aladdin\_cmr\_la/src_

## Folder: src
### > src/run_training.py

The main file to train the image registration networks (Aladdin-R and VoxelMorph; Default: Aladdin-R) to obtain the displacement vector fields.

Parameters of the models are taken from the appropriate _configuration/configuration\_\<model\>.py_ file. The best weights of the models are saved at _../checkpoint/\<model\>_\<parameters\>_\<date\>_. Note: the expected root folder when executing _run\_training.py_ is _src_.

### > src/run_data_caching.py

Optional script to preprocess the dataset and saves it to cache. This process is included in the  _run\_training.py_. Cached data is stored at _../data\_\<parameters\>\_cache_

### > src/run_output_generation.py

Saves the estimted displacement vector fields and transformed segmenatation maps of the models trained with _run\_training.py_. The results are stored at _../outputs\_/\<model\>_.


## Folder: src/configuration
### > src/configuration/configuration_aladdin_r.py

Model configration for Aladdin-R.

### > src/configuration/configuration_vxm.py

Model configration for VoxelMorph.

### > src/configuration/configuration_vxmseg.py

Model configration for VoxelMorprh with segmentation map inputs.

### > src/configuration/utils.py

Contains miscellaneous functions for the configration.


## Folder: src/loader
### > src/loader/data_generator.py

Functions for loading, caching, and preprocessing the data and providing a generator for the data.

### > src/loader/tf_generator.py

A TensorFlow wrapper for the data generator defined in _src/loader/data\_generator.py_.


## Folder: src/tf
### > src/tf/callbacks/callback.py

Contains the collection of Keras/TensorFlow callbacks used during training.

### > src/tf/callbacks/weights.py

Defines the Keras/TensorFlow callbacks that are responsible for (re-)loading the model's weights during training.

### > src/tf/losses/deform.py

Defines loss functions used for the image registration networks.

### > src/tf/losses/loss.py

Defines the L1/L2 losses.

### > src/tf/metrics/metrics.py

Defines the metrics to estimate during training.

### > src/tf/models/aladdin_r.py

Defines Aladdin-R's TensorFlow architecture.

### > src/tf/models/voxelmorph.py

Defines VoxelMorph's and VoxelMorph-Seg's TensorFlow architecture.

### > src/tf/utils/load.py

Contains miscellaneous functions responsible for load the TensorFlow models.

### > src/tf/utils/seed.py

Contains function for setting the global random number generator seed.


## Folder: src/atlas_construction
### > src/atlas_construction/atlas_generation.py

(Step 1) Constructs the deformation atlas from the healthy cases found by default at _../../data\_nn/train_. Healthy cases are cases which do not have the prefix of '_PAT_'. The atlas mesh (.vtk) is saved at _src/atlas_construction/\_atlas\_output_.

Note: functions in the _atlas\_construction_ expect root folder to be _atlas\_construction_.

### > src/atlas_construction/register_to_atlas.py

(Step 2) Register each case to the atlas. By default expects the atlas to be located at _atlas_construction/\_atlas\_output_ and the cases to register at _../../data\_nn/train_. The per case registration output is saved at _atlas_construction/\_registration\_output_.

### > src/atlas_construction/atlas_stats.py

(Step 3) Estimates the the mean (key: 'mean'), standard deviation ('std'), coefficient of variation ('coef'), and covariance ('cov') of the displacement vector fields ('dvf') and strains ('Epr'). By default, expects the atlas to be located at _atlas_construction/\_atlas\_output_ and the healthy cases at _atlas_construction/\_registration\_output_. The statistics of the atlas are saved at _atlas_construction/\_atlas\_stats\_output_. 

### > src/atlas_construction/registration_stats.py

(Step 4) Calculates the Mahalanobis distance between an individual case and the atlas' dsitribution. Expects by default the statistical deformation atlas to be located at _atlas_construction/\_atlas\_stats\_output_ and the individual cases at _atlas_construction/\_registration\_output_. The results across the cardiac cycle are saved at _atlas_construction/\_registration\_stats\_output_.
