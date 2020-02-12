![fluffy](./data/fluffy.jpg)

Fluffy
==============================

Reproducible deep learning based segmentation of biomedical images.



## Overview

- [Project Organization](#project-organization)
- [Why use my workflow?](#why-use-my-workflow)
- [Results](#results)
- [System requirements and installation](#system-requirements-and-installation)
- [Data and model availability](#data-and-model-availability)
- [Training and inferencing](#training-and-inferencing)
- [Roadmap](#roadmap)



### Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train` (to follow)
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data               <- Only for README or notebook access. For training data read below.
    │
    ├── docs               <- Currently home to the manual.
    │
    ├── notebooks          <- Jupyter notebooks. The naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── environment.yml    <- The conda environment file to reproduce the conda environment; see below
    ├── requirements.txt   <- The requirements file for reproducing the virtualenv environment; see github.com/pypa/virtualenv
    │
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
    │   ├── tests          <- Scripts to test all other functionality using pytest
    │       └── test_train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org



### Why use my workflow?

My proposed workflow is still in development and will be improved as we speak. Here a quick rundown of the main features:

- Only well-maintained packages used (numpy, pandas, tensorflow, scikit-image, opencv-python)
- Extensive testing to make sure no error gets left unchecked
- Simple usage
- Not so bad results almost guaranteed



### Results

* Nuclear segmentation using the categorical model providing a class to separate nuclei. See [here](./data/example_nucleus.pdf).

* Granular segmentation illustrating the selectivity of the model. See [here](./data/example_granules.pdf).



### System requirements and installation

This workflow uses two main elements:

1. Code in this repository
2. Conda package manager (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))


Set up your environment using the following commands:

```bash
git clone -b v1 https://github.com/bbquercus/fluffy
cd fluffy

# Conda
conda env create -f environment.yml
conda activate fluffy

# Pip
pip install -r requirements.txt
```



### Data and model availability

Data is currently not available but all annotated images will be released after enough testing was performed. Once published, the makefile will automatically download data to the `data` directory. Pretrained models can be found [here](https://www.dropbox.com/sh/5ffku4w4n52urbj/AADAACaMf3wEDyNfWOjdi9BOa?dl=0).



### Training and inferencing

For a detailed guide, please read the extensive [manual](https://github.com/bbquercus/fluffy/manual.pdf). For pythonistas among you - training information will be added shortly. For inference please call:

```bash
cd src/models/
python predict_model.py --model_file='path_to_model.h5' --image='path_to_folder_with_images'
```



### Roadmap

- [ ] Fully automated luigi pipeline for model training
- [ ] Streamlit application for easy inferencing
- [ ] Open sourcing of all training data and models
- [ ] Makefile to download models
- [ ] Addition of spot detection (in colaboration with @zhanyinx)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience <br/> Title image designed by <a href="http://www.freepik.com">vectorpocket / Freepik</a></small></p>
