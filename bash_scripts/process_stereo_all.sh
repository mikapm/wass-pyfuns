#!/bin/bash

# Script to run all WASS processing steps sequentially:
# Processing, mean-plane estimation and gridding
# Command line input arguments: 
# 1. Path to root directory
# 2. Number of frames to include
# 3. Number of cores to parallelize processing with
# 4. Grid resolution
# 5. and higher: date-time strings (format: yyyymmddHHMM)

rootdir="$1"
nframes="$2"
ncores="$3"
dxy="$4"

# set -- "{@:4:$#-1}"

for datetime in "${@:5}"; do

    # Set inputs #
    d=${datetime:0:8} # date string
    t=${datetime:8:13} # time string
    expdir="$rootdir"/"$d"/"$t" # Experiment directory
    echo $d
    echo $t
    echo $expdir

    # Run the frame renaming #
    python wass-pyfuns/pywass/path/rename_frames.py -nf $nframes -date $d -time $t

    # Run the processing #
    python wass-pyfuns/pywass/wass_launch.py -dr $expdir -pc $ncores -wr /home/wilsongr-local/wass

    # Run the gridding #
    python wass-pyfuns/pywass/mean_plane.py -dr $expdir -baseline 5.51 -ymin -150 -ymax -100 -xmin -25 -xmax 25
    python wass-pyfuns/pywass/mesh_to_ncgrid_v2.py -dr $expdir -dxy $dxy -xmin -60 -xmax 70 -ymin -180 -ymax -60

    # Delete the input directory that was created, which contains a tif version of every image for both cams #
    # rm -rf "${expdir}/input"

    # Make a softlink to the nc file in the dist directory #
    # Note: if no /grid/wass.nc file was made, the datetime is skipped, won't show up in dist
    if test -f "$expdir"/grid/wass.nc; then
	distdir="path/stereo/dist"
	ln -rs "$expdir"/grid/wass.nc "$distdir"/"$d"_"$t"_wass.nc 
    fi

done
