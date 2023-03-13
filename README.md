# wass-pyfuns
Python functions to run the [WASS stereo wave processing](https://github.com/fbergama/wass "WASS Github repository") executables, and load and post process the WASS output files.

## General
WASS (Waves Acquisition Stereo System) is an open source pipeline for 3D reconstruction of wave-field stereo imagery. The codes in this repository are not related to the official WASS distribution in any way.

Links to [WASS Github repository](https://github.com/fbergama/wass) and [WASS website](https://www.dais.unive.it/wass/).

Most of the codes in this repository are based on the Matlab scripts included by default in the WASS installation. The codes were originally written to facilitate the processing of stereo imagery acquired from the Ekofisk stereo video observatory as part of the StereoWave research project of the Norwegian Meteorological Institute (MET Norway). Therefore some functionality (mainly related to image pre-processing) is relevant to this specific data set only. 

These codes have been tested in a Linux environment. The bash_scripts folder includes scripts and specific instructions on how to run WASS processing in the computing backend of the MET Norway post-processing infrastructure (PPI).

## Installation
```
$ mamba env create -f environment.yml
$ conda activate wass
```

## Update environment
```
mamba env update --file environment.yml --prune
```

## Dependencies
The WASS executables have to be installed to run the stereo processing pipeline. The installation instructions are found on the WASS website linked above.

The functions to run the executables and load processed output depend on basic python libraries for scientific data analysis such as numpy and scipy as well as the OpenCV image processing library (cv2).

## Overview
Below is an overview and brief descriptions of the files in this repository.

### Image pre-processing (Ekofisk-specific)
 - `prep_images.py` Converts Ekofisk binary raw images to TIF format and renames the images according to WASS conventions.
 - `match_raw_images.py` Checks for inconsistencies (e.g. unsynced frame pairs or missing frames) in stereo frame pairs.

### Running WASS
 - `wass_launch.py` Run the WASS executables (tested in a Linux environment). Option to parallelize the stereo-frame-wise processing using the subprocess python library.

### Loading and post-processing WASS output
 - `mean_plane.py` Estimate the mean sea plane from the 3D point cloud over a selected region of the stereo footprint.
 - `wass_load.py` Read a WASS output mesh (i.e. 3D point cloud), and rotate and align it according to the mean sea plane. Also includes functionality to interpolate the point cloud to a regular 2D grid.
 - `mesh_to_ncgrid.py` Generate netcdf files with time-sequences of consecutive 2D grids.
