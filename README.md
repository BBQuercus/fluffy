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
    ├── environment_gpu.yml <- Same conda environment file for gpus
    ├── requirements.txt   <- The requirements file for reproducing the virtualenv environment; see github.com/pypa/virtualenv
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to format, preprocess, etc. labeled data
    │   │   ├── make_directories.py
    │   │   ├── make_labelling.py
    │   │   ├── make_masks.ijm
    │   │   └── make_label_maps.py
    │   │
    │   ├── training       <- Scripts to train models models
    │   │   ├── utils.py
    │   │   └── train_model.py
    │   │
    │   ├── inference      <- Scripts to use the trained models
    │   │   ├── predict_model.py
    │   │   └── fluffy.py
    │   │
    │   └── testing        <- Scripts to test all other functionality using pytest (to follow)
    │       └── test_train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org



### Why use my workflow?

My proposed workflow is still in development and will be improved as we speak. Here a quick rundown of the main features:

- Only well-maintained packages used (numpy, tensorflow, scikit-image, opencv-python)
- Extensive testing to make sure no error gets left unchecked
- Simple usage
- Not so bad results almost guaranteed



### Results

* Nuclear segmentation using the categorical model providing a class to separate nuclei. See [here](./data/example_nucleus.pdf).

* Granular segmentation illustrating the selectivity of the model. See [here](./data/example_granules.pdf).



### System requirements and installation

This workflow has two main functions:

* Training your own models
* Inferencing via the fluffy interface

These two parts have different requirements:

```bash
# For training
git clone -b v1 https://github.com/bbquercus/fluffy
cd fluffy
conda env create -f environment.yml
conda activate fluffy

# For inferencing
docker pull bbquercus/fluffy
docker run -p 8501:8501 -v /FOLDER/:/FOLDER/ bbquercus/fluffy
# Substitute FOLDER with the root file directory
# Visit localhost:8501
```



### Data and model availability

Data is currently not available but all annotated images will be released after enough testing was performed. Once published, the makefile will automatically download data to the `data` directory. Pretrained models are automatically downloaded within the streamlit interface.



### Training and inferencing

Please read the extensive [manual](https://github.com/bbquercus/fluffy/docs/manual.pdf).



### Roadmap

- [x] Streamlit application for easy inferencing
- [x] Fully automated metaflow pipeline for model training
- [ ] Open sourcing of all training data and models
- [ ] Makefile to download models
- [ ] Addition of spot detection (in colaboration with @zhanyinx)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience <br/> Title image designed by <a href="http://www.freepik.com">vectorpocket / Freepik</a>.</small></p>
