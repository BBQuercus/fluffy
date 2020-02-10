

fluffy-guide
==============================

Reproducible deep learning based segmentation of biomedical images.



## Overview

- [Project Organization](#project-organization)
- [Why use my workflow?](#why-use-my-workflow)
- [System requirements and installation](#system-requirements-and-installation)
- [Data availability](#data-and-model-availablilty)
- [Training and inferencing](#training-and-inferencing)



### Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train` (to follow)
    ├── README.md          <- The top-level README for developers using this project.
    ├── data (to follow, see below)
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details (to follow, for now, use the manual)
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries (see below)
    │
    ├── notebooks          <- Jupyter notebooks. The naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting (to follow)
    │
    ├── environment.yml    <- The conda environment file to reproduce the conda environment; see below
    ├── requirements.txt   <- The requirements file for reproducing the virtualenv environment; see github.com/pypa/virtualenv
    │
    ├── manual.pdf         <- A extensive manual covering all topics of this workflow
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported (to follow)
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to format, preprocess, etc. labeled data
    │   │   ├── make_directories.py
    │   │   ├── make_labelling.py
    │   │   └── make_label_maps.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── tests       <- Scripts to test all other functionality using pytest
    │   │   └── test_train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations (to follow, currently as notebook)
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org (to follow, currently in src/tests)



### Why use my workflow?

My proposed workflow is still in development and will be improved as we speak. Here a quick rundown of the main features:

- Only well-maintained packages used (numpy, pandas, tensorflow, scikit-image, opencv-python)
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



### Data and model availability

Data is currently not available but all annotated images will be released after enough testing was performed. Once published, the makefile will automatically download data to the `data` directory. Pretrained models can be found [here](https://www.dropbox.com/sh/5ffku4w4n52urbj/AADAACaMf3wEDyNfWOjdi9BOa?dl=0).



### Training and inferencing

For a detailed guide, please read the extensive [manual](https://github.com/bbquercus/fluffy-guide/manual.pdf). For pythonistas among you - training information will be added shortly. For inference please call:

```bash
cd src/models/
python predict_model.py --model_file='path_to_model.h5' --image='path_to_folder_with_images'
```



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
