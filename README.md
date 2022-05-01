# DLFAV-PROJECT

## Introduction
This project is carried out as part of the Deep Learning for Autonomous Vehicles course given by Alexander Alahi.

This project combines the [OpenPifPaf](https://openpifpaf.github.io/) package for pose estimation, and a [YOLOX DeepSORT tracker](https://github.com/pmj110119/YOLOX_deepsort_tracker) for tracking with ReID capabilities. 

A pose detector composed of a simple multilayer perceptron classifies the output of OpenPifPaf from a predefied set of poses, which is then used by a sequence detector for the selection of a person of interest.

## Installation 

In a command window, run the following:

```bash
git clone --recurse-submodules https://github.com/tgieruc/DLFAV-PROJECT
cd DLFAV-PROJECT
pip install -r requirements.txt
```

If an error pops up, try manually installing the dependencies in the requirements.txt file. Python 3.9 is recommended for all packages to work properly.

## Testing 

Run the `milestone_2.ipynb` notebook to test the program.

