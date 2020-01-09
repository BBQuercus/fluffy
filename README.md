# Deep learning workflow


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

1. This repository
2. Docker as container providing all dependencies

First, clone this repository by running
```
git clone -b v1 https://github.com/bbquercus/reimagined_disco
cd reimagined_disco
```

Once Docker is installed (see [here](https://docs.docker.com/install/)), run `docker build -t bbquercus:v1 .` in the directory containing a clone of this repository. Activate the local environment through `docker run -i -t bbquercus:v1 /bin/bash`.



### Data availability

All training and testing data (not including raw files) can be found as `.npy` files [here](link). Please place these files into the `data/processed/` directory for the workflow to function properly.



### Training the model

Once the local environment was set up according to the instructions above, the model can be trained by calling one of the following calls:
- `python src/main.py` – a full hyperparameter search, cross validation and report generation based on the available data
- `python src/main.py --best_only` – reproduce the best model only



### Inferencing with existing model

The final model is available in the `models/` directory. Inferencing can be done as follows:

* `python src/inference.py --indir INPUT_DIRECTORY --outdir OUTPUT_DIRECTORY`
* The INPUT\_DIRECTORY should contain files in one of the following formats: .tif, .tiff, .jpeg, .png, .stk, be two dimensional only (one single Z-stack for example).
* The OUTPUT\_DIRECTORY will be created if not yet existant.
