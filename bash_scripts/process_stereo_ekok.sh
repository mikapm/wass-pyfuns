#!/bin/bash

# Script to run all WASS processing steps sequentially:
# Processing, mean-plane estimation and gridding
# Command line input arguments: 
# 1. Path to root directory
# 2. Number of cores to parallelize processing with
# 3. and higher: date-time strings (format: yyyymmddHHMM)

# Example (parallelized with 6 cores, 2022/09/16 16:00-16:20):
# ./process_stereo_ekok.sh /path/to/experiment_root/ 6 202209161600

rootdir="$1"
ncores="$2"

# Loop over requested datetimes
for datetime in "${@:3}"; do

    # Set inputs #
    d=${datetime:0:8} # date string
    t=${datetime:8:13} # time string
    expdir_base="$rootdir"/"$d" # Experiment date directory
    expdir="$expdir_base"/"$t" # Experiment time directory
    echo " "
    echo "date: ""$d"
    echo "start time: ""$t"
    echo "Experiment directory: ""$expdir"
    echo " "

    # Check if expdir_base exists
    if [ ! -d "$expdir_base" ]; then
        echo "$expdir_base does not exist, generating it ..."
        mkdir "$expdir_base"
    fi
    # Check if expdir exists 
    if [ ! -d "$expdir_base" ]; then
        echo "$expdir_base does not exist, generating it ..."
        mkdir "$expdir_base"
    fi

    # Copy over config folder (assuming it exists in $rootdir)
    configdir="$expdir"/config
    # Check if config dir. exists 
    if [ ! -d "$configdir" ]; then
        echo "Copying over config folder to ""$configdir"
        cp -r "$rootdir"/config "$expdir"/config
    fi

    # Check if input image dir. (for WASS) exists
    inputdir="$expdir"/input
    if [ ! -d "$inputdir" ]; then
        echo "$inputdir does not exist, generating it ..."
        mkdir "$inputdir"
    fi
    # Run file conversion raw -> tif (assuming raw frames in expdir/raw folder)
    echo " "
    echo "Converting frames ..."
    # python /home/mikapm/Github/wass-pyfuns/pywass/prep_images.py "$expdir"/raw -outdir $inputdir

    # Run the processing #
    echo " "
    echo "Running WASS ..."
    python /home/mikapm/Github/wass-pyfuns/pywass/wass_launch.py -dr "$expdir" -pc "$ncores"

    # Calculate mean planes. Use batches to run in parallel #
    echo " "
    echo "Computing mean planes ..."
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 0 -ind_e 125 &
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 126 -ind_e 250 &
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 251 -ind_e 375 &
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 376 -ind_e 500 &
    wait # This will wait until all above scripts finish
    # Average planes
    echo " "
    echo "Averaging planes ..."
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 376 -ind_e 500 --avg_planes

    # # Run the gridding #
    # echo " "
    # echo "Running gridding ..."
    # python wass-pyfuns/pywass/mesh_to_ncgrid_v2_ekok.py -dr $expdir -dxy $dxy -xmin -60 -xmax 70 -ymin -180 -ymax -60

    # Delete the input directory that was created, which contains a tif version of every image for both cams #
    # rm -rf "${expdir}/input"

    # Make a softlink to the nc file in the dist directory #
    # Note: if no /grid/wass.nc file was made, the datetime is skipped, won't show up in dist
    # if test -f "$expdir"/grid/wass.nc; then
    #     distdir="path/stereo/dist"
    #     ln -rs "$expdir"/grid/wass.nc "$distdir"/"$d"_"$t"_wass.nc 
    # fi

done
