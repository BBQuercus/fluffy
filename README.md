

fluffy-guide
==============================

Reproducible deep learning based segmentation of biomedical images.



## Overview

- Project Organization
- Why use my workflow?
- System requirements and installation
- Data availability
- Training the model
- Inferencing with existing model



### Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train` (to follow)
    ├── README.md          <- The top-level README for developers using this project.
    ├── data (to follow, see below)
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries (to follow, see below)
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials. (to follow)
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting (to follow)
    │
    ├── environment.yml    <- The conda environment file to reproduce the conda environment; see below
    ├── requirements.txt   <- The requirements file for reproducing the virtualenv environment; see github.com/pypa/virtualenv
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported (to follow)
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py (to follow)
    │   │   └── train_model.py
    │   │
    │   ├── tests       <- Scripts to test all other functionality using pytest
    │   │   └── test_train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations (to follow, currently as notebook)
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org (to follow, currently in src/tests)



### Why use my workflow?

My proposed workflow is still in development and will be improved as we speak. Here a quick rundown of the main features:

- Only well maintained packages used (numpy, pandas, tensorflow, scikit-image, opencv-python)
- Extensive testing to make sure no error gets left unchecked
- Simple usage
- Not so bad results almost guaranteed ^^



### System requirements and installation

This workflow uses two main elements:

1. Code in this repository
2. Conda package manager (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))


Set up your environment using the following commands:

```bash
git clone -b v1 https://github.com/bbquercus/fluffy-guide
cd fluffy-guide

# Conda
conda env create -f environment.yml
conda activate fluffy-guide

# Pip
pip install -r requirements.txt
```



### Data availability

Data is currently not available but all annotated images will be released after enough testing was performed. Once published, the makefile will automatically download data to the `data` directory.



### Training the model

Once the local environment was set up according to the instructions above, the model can be trained by calling one of the following calls:

```bash
cd src/models/
# Binary model
python train_model.py --type='binary'
# Categorical model
python train_model.py --type='categorical'
```



### Inferencing with existing model

All pretrained models used in the inference notebook can be found [here](https://www.dropbox.com/sh/5ffku4w4n52urbj/AADAACaMf3wEDyNfWOjdi9BOa?dl=0). Follow along the notebook `1.0-be-inference.ipynb` to test one of the models.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

