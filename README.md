Fluffy-Guide
==============================

Pipeline for segmenting biomedical microscopy images.


### Overview

- Why use my workflow?
- System requirements and installation
- Data availability
- Training the model
- Inferencing with existing model



### Why use my workflow?

Information what my super duper thing can do...



### System requirements and installation

This workflow uses two main elements:

1. Code in this repository
2. Conda package manager (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))


Set up your environment using the following commands:

```bash
git clone -b v1 https://github.com/bbquercus/fluffy-guide
cd fluffy-guide
conda env create -f environment.yml
conda activate tf
```



### Data availability

All training and testing data (not including raw files) can be found as `.npy` files [here](link). Please place these files into the `data/processed/` directory for the workflow to function properly.



### Training the model

Once the local environment was set up according to the instructions above, the model can be trained by calling one of the following calls:

```bash
# Binary model
python train_binary.py
# Categorical model
python train_categorical.py
```



### Inferencing with existing model

The final model is available in the `models/` directory. Inferencing can be done as follows:

* `python src/inference.py --indir INPUT_DIRECTORY --outdir OUTPUT_DIRECTORY`
* The INPUT\_DIRECTORY should contain files in one of the following formats: .tif, .tiff, .jpeg, .png, .stk, be two dimensional only (one single Z-stack for example).
* The OUTPUT\_DIRECTORY will be created if not yet existant.
