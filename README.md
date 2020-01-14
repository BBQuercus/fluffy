# Deep learning workflow


### Overview
- Development plan
- Why use my workflow?
- System requirements and installation
- Data availability
- Training the model
- Inferencing with existing model



### Development plan

* Platform and framework
  - [ ] Tensorflow 2.x / Keras
  - [ ] macOS
  - [ ] Ubuntu
* Explore model architecture
  - [ ] UNet
    - [ ] Contour Aware model (2 tasks)
    - [ ] Contour Aware Marker model (3 tasks)
    - [ ] Boundaries detection for adjacent nuclei only
  - [ ] Mixed-Scale Dense CNN (super slow training/inference on current deep learning framework design)
  - [ ] Dilated Convolution
  - [ ] Dropout
  - [ ] Batch normalization
  - [ ] Transfer learning
    - [ ] Vgg (16)
    - [ ] ResNet (34, 101)
    - [ ] DenseNet (121, 201)
  
  - [ ] Score function
  - [ ] Cost functions
    + [ ] binary cross entropy
    + [ ] pixel wise IoU, regardless of instances
    + [ ] loss weight per distance of instances's boundary
    + [ ] Focal loss (attention on imbalance loss)
    + [ ] Distance transform based weight map
    + [ ] Shape aware weight map
  
* Hyper-parameter tunning
  - [ ] Learning rate
  - [ ] Input size (Tried 384x384, postponed due to slow training pace)
  - [ ] Confidence level threshold
    - [ ] A better way to tune these gating thresholds
  - [ ] Evaluate performance of mean and std of channels
* Data augmentation
  - [ ] Random crop
  - [ ] Random horizontal and vertical flip
  - [ ] Random aspect resize
  - [ ] Random color adjustment
  - [ ] Random color invert (NG for this competition)
  - [ ] Random elastic distortion
  - [ ] Contrast limited adaptive histogram equalization (NG for this competition)
  - [ ] Random rotate
  - [ ] Random noise (additive gaussian and multiplied speckle) (NG for this competition)
  - [ ] Random channel shuffle and color space transform
* Dataset
  - [ ] Support multiple whitelist filters to select data type
  - [ ] Support manually oversample in advance mode
  - [ ] Auto-balance data distribution weight via oversampling
  - [ ] Tools to crop the portion we need
* Pre-process
  - [ ] Input normalization
  - [ ] Binarize label
  - [ ] Cross-validation split
  - [ ] Verify training data whether png masks aligned with cvs mask.
  - [ ] Blacklist mechanism to filter noisy label(s)
  - [ ] Annotate edge as soft label, hint model less aggressive on nuclei edge
  - [ ] Whitelist configure option of sub-category(s) for training / validation
  - Prediction datafeed (aka. arbitrary size of image prediction)
    + [ ] Resize and regrowth
    + [ ] Origin image size with border padding (black/white constant color)
    + [ ] Origin image size with border padding (replicate border color)
    + [ ] Tile-based with overlap
  - [ ] Easy and robust mechanism to enable incremental data addition & data distribution adjustment
  - [ ] 'Patient' level CV isolation, not infected by data distribution adjustment
  - [ ] Convert input data to CIELAB color space instead of RGB
  - [ ] Use [color map algorithm](https://stackoverflow.com/questions/42863543/applying-the-4-color-theorem-to-list-of-neighbor-polygons-stocked-in-a-graph-arr) to generate ground truth of limited label (4-), in order to prevent cross-talking
* Post-process
  - Watershed segmentation group
    + [ ] Marker by statistics of local clustering peak
    + [ ] Marker by contour-based from model prediction
    + [ ] Marker by marker-based from model prediction
  - Random walker segmentation group
    + [ ] Marker by statistics of local clustering peak
    + [ ] Marker by contour-based from model prediction
    + [ ] Marker by marker-based from model prediction
  - Ensemble
    + [ ] Average probability of pixel wise output of multiple models (or checkpoints)
  - Test Time Augmentation
    + [ ] Horizontal flip, vertical flip, and combined views
    + [ ] RGB to grayscale (NG for this competition)
    + [ ] Rotate 90/180/270 degree views
  - [ ] Fill hole inside each segment group
* Computation performance
  - [ ] CPU
  - [ ] GPU
  - [ ] Multiple subprocess workers (IPC)
  - [ ] Cache images
  - [ ] Redundant extra contour loop in dataset / preprocess (~ 50% time cost)
  - [ ] Parallel CPU/GPU pipeline, queue or double buffer
* Statistics and error analysis
  - [ ] Mini-batch time cost (IO and compute)
  - [ ] Mini-batch loss
  - [ ] Mini-batch IOU
  - [ ] Visualize prediction result
  - [ ] Visualize log summary in TensorBoard
  - [ ] Running length output
  - [ ] Graph visualization
  - [ ] Enhance preduction color map to distinguish color of different nucleis
  - [ ] Visualize overlapping of original and prediction nucleis
  - [ ] Statistics of per channel data distribution, particular toward alpha
  - [ ] Auto save weight of best checkpoint, IoU of train and CV, besides period save.



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

Once Docker is installed (see [here](https://docs.docker.com/install/)) and launched, run...
`docker build -t bbquercus:v1 .`
...in the directory containing a clone of this repository. Activate the local environment through `docker run -i -t bbquercus:v1 /bin/bash`.



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
