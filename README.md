# wass-pyfuns
Python functions to run the [WASS stereo wave processing](https://github.com/fbergama/wass "WASS Github repository") executables, and load and post process the WASS output files.

## General
WASS (Waves Acquisition Stereo System) is an open source pipeline for 3D reconstruction of wave-field stereo imagery.

Links to [WASS Github repository](https://github.com/fbergama/wass) and [WASS website](https://www.dais.unive.it/wass/).

These codes are based on the Matlab scripts included by default in the WASS installation. The codes were originally written to facilitate the processing of stereo imagery acquired from the Ekofisk stereo video observatory, therefore some functionality (mainly related to image pre-processing) is relevant for this specific data set only.

## Dependencies
The WASS executables have to be installed to run the stereo processing pipeline. The installation instructions are found on the WASS website linked above.

The functions to run the executables and load processed output depend on basic python libraries for scientific data analysis such as numpy and scipy.

## Overview
Below is an overview and brief descriptions of the files in this repository.

### Image pre-processing (Ekofisk-specific)
 - `match_raw_images.py`
 - `prep_images.py` 

### Running WASS
 - `wass_launch.py`

