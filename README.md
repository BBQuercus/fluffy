![fluffy](./data/fluffy.png)

Fluffy
==============================

Reproducible deep learning based segmentation of biomedical images.



## Overview

- [What is fluffy?](#what-is-fluffy)
- [Project organization](#project-organization)
- [Examples](#examples)
- [System requirements and installation](#system-requirements-and-installation)
- [Data and model availability](#data-and-model-availability)
- [Labeling and data preparation](#labeling-and-data-preparation)
- [Roadmap](#roadmap)
- [Citation](#citation)



### What is fluffy?

Fluffy is a simple browser based tool to use custom deep learning models for biomedical image segmentation.
As microscopy images are usually large files, a local docker container is used.
In comparison to a standard web server this greatly reduces file transfer and speeds everything up.

Some key features available include:
* A couple of models available
    * Nuclear segmentation
    * Cytoplasmic segmentation
    * Stress granule segmentation
    * ER segmentation
* Single image viewing to check how good the models are
* Batch processing to process multiple files at once

Additionally, all code is relying on well-maintained packages (numpy, tensorflow, scikit-image, flask).



### Project organization

This repository:
```
    ├── LICENSE
    ├── README.md          <- This top-level README.
    ├── data/              <- Sample data to be displayed. For training data read below.
    ├── docs/              <- Home to the manual.
    └── web/               <- Fluffy interface. Flask application used in the docker container.
```
Aditionally:
* [DockerHub](hub.docker.com/r/bbquercus/fluffy)
* [Training repository](https://github.com/bbquercus/)



### Examples

* Nuclear segmentation using the categorical model providing a class to separate nuclei. See [here](./data/example_nucleus.pdf).
* Granular segmentation illustrating the selectivity of the model. See [here](./data/example_granules.pdf).



### System requirements and installation

Go to [DockerHub](hub.docker.com/r/bbquercus/fluffy) and find the latest version of fluffy.

```bash
# Replace with the latest version at hub.docker.com/r/bbquercus/fluffy  
docker pull bbquercus/fluffy:VERSION
docker run -p 5000:5000 bbquercus/fluffy:VERSION
```

Visit localhost:5000 in your browser of choice.



### Data and model availability

Data is currently not available but all annotated images will be released after enough testing was performed.
Pretrained models are automatically downloaded within the interface or can be accessed [here](https://drive.google.com/open?id=1dSD8zS3POp1SV1iJ8mPj9qIOFZT0ClR9).



### Labeling and data preparation

Labeling is done in Fiji and data preparation using simple command line tools within a conda environment.
Both processes are described in the extensive [manual](https://github.com/bbquercus/fluffy/docs/manual.pdf).
Training must be done at one's own risk or by asking me.
The training is also open sourced [here](https://github.com/bbquercus/).



### Roadmap

- [x] Flask application for easy inferencing.
- [x] Separate training from inference. Fluffy will only remain for inference via the flask application.
- [ ] Open sourcing of all training data.
- [ ] Addition of spot detection (in collaboration with @zhanyinx).



### Citation

If you find fluffy to be useful, please cite my repository:

```
@misc{Fluffy,
      author = {Bastian Th., Eichenberger},
      title = {Fluffy},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/bbquercus/fluffy}}
```