![fluffy](./data/fluffy_header.png)

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
    ├── data/              <- Only for README or notebook access. For training data read below.
    │
    ├── docs/              <- Home to the manual.
    │
    ├── Dockerfile         <- Dockerfile for the fluffy interface.
    ├── environment.yml    <- The conda environment file to reproduce the conda environment; see below
    ├── environment_gpu.yml <- Same conda environment file for GPUs
    ├── requirements.txt   <- The requirements file for reproducing the virtualenv environment; see github.com/pypa/virtualenv
    │
    ├── src/               <- Source code for use in this project.
    │   │
    │   ├── data/          <- Scripts to format, preprocess, etc. labeled data
    │   │   ├── make_directories.py
    │   │   ├── make_labelling.py
    │   │   ├── make_masks.ijm
    │   │   └── make_label_maps.py
    │   │
    │   ├── training/      <- Scripts to train models models
    │   │   ├── config.py
    │   │   ├── data.py
    │   │   ├── dirtools.py
    │   │   ├── metrics.py
    │   │   ├── models.py
    │   │   ├── train_model.py <- Fully automated pipeline
    │   │   └── train_simple.py <- Simple script for quick prototyping
    │   │
    │   ├── inference/     <- CLI to use the trained models
    │   │   └── predict_model.py
    │   │
    │   └── testing/       <- Scripts to test all other functionality using pytest (to follow)
    │       └── test_train_model.py
    │
    └── web/               <- Fluffy interface. Flask application used in the docker container.



### Why use my workflow?

My proposed workflow is still in development and will be improved as we speak. Here a quick rundown of the main features:

- Only well-maintained packages used (numpy, tensorflow, scikit-image)
- Extensive testing to make sure no error gets left unchecked
- Simple usage via the fluffy interface
- Not so bad results almost guaranteed



### Results

* Nuclear segmentation using the categorical model providing a class to separate nuclei. See [here](./data/example_nucleus.pdf).

* Granular segmentation illustrating the selectivity of the model. See [here](./data/example_granules.pdf).



### System requirements and installation

For training and CLI based inferencing:
```bash
git clone -b v1 https://github.com/bbquercus/fluffy
cd fluffy
conda env create -f environment.yml
conda activate fluffy
```

To use the fluffy interface:
```bash
# Replace with the latest version at hub.docker.com/r/bbquercus/fluffy  
docker pull bbquercus/fluffy:VERSION
docker run -p 5000:5000 bbquercus/fluffy:VERSION
# Visit localhost:5000
```



### Data and model availability

Data is currently not available but all annotated images will be released after enough testing was performed. Once published, the makefile will automatically download data to the `data` directory. Pretrained models are automatically downloaded within the interface.



### Training and inferencing

To inference, a fluffy interface is available through docker.
Training and CLI based inferencing can be done within a conda environment.
Both options are described in the extensive [manual](https://github.com/bbquercus/fluffy/docs/manual.pdf).



### Roadmap

- [x] Flask application for easy inferencing
- [x] Fully automated metaflow pipeline for model training
- [ ] Open sourcing of all training data and models
- [ ] Makefile to download models
- [ ] Addition of spot detection (in collaboration with @zhanyinx)
