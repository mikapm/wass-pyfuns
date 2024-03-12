#!/bin/bash

# Script to run all WASS processing steps sequentially:
# Processing, mean-plane estimation and gridding
# Command line input arguments: 
# 1. Path to root directory
# 2. Number of cores to parallelize processing with
# 3. and higher: date-time strings (format: yyyymmddHHMM)

# Example (parallelized with 6 cores):
# ./process_stereo_ekok.sh /path/to/experiment_base/ 6

rootdir="$1"
ncores="$2"

# Loop over requested datetimes
for datetime in "${@:3}"; do

    # Set inputs #
    d=${datetime:0:8} # date string
    t=${datetime:8:13} # time string
    expdir="$rootdir"/"$d"/"$t" # Experiment directory
    echo $d
    echo $t
    echo $expdir

    # Run file conversion raw -> tif . Use batches to run in parallel #
    echo "Converting frames ..."
    python wass-pyfuns/pywass/prep_images.py /path/to/raw -outdir $expdir

    # Run the processing #
    echo "Running WASS ..."
    python wass-pyfuns/pywass/wass_launch.py -dr $expdir -pc $ncores -wr /home/local/wass

    # Calculate mean planes. Use batches to run in parallel #
    echo "Computing mean planes ..."
    python wass-pyfuns/pywass/mean_plane.py -dr $expdir -ind_s 0 -ind_e 500 &
    python wass-pyfuns/pywass/mean_plane.py -dr $expdir -ind_s 501 -ind_e 1000 &
    python wass-pyfuns/pywass/mean_plane.py -dr $expdir -ind_s 1001 -ind_e 1500 &
    wait # This will wait until all scripts finish

    # Run the gridding #
    echo "Running gridding ..."
    python wass-pyfuns/pywass/mesh_to_ncgrid_v2_ekok.py -dr $expdir -dxy $dxy -xmin -60 -xmax 70 -ymin -180 -ymax -60

    # Delete the input directory that was created, which contains a tif version of every image for both cams #
    # rm -rf "${expdir}/input"

    # Make a softlink to the nc file in the dist directory #
    # Note: if no /grid/wass.nc file was made, the datetime is skipped, won't show up in dist
    # if test -f "$expdir"/grid/wass.nc; then
    #     distdir="path/stereo/dist"
    #     ln -rs "$expdir"/grid/wass.nc "$distdir"/"$d"_"$t"_wass.nc 
    # fi

done
