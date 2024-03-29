#!/bin/bash
#
# Script to run all WASS processing steps sequentially:
# Image conversion, processing, mean-plane estimation and gridding
# Command line input arguments: 
# 1. Path to root experiment directory
# 2. Number of cores to parallelize processing with
# 3. and higher: date-time strings (format: yyyymmddHHMM)
#
# Example (parallelized with 6 cores, 2022/09/16 16:00-16:20):
# ./process_stereo_ekok.sh /path/to/experiment_root/ 6 202209161600
#
# Note that the mean plane estimation and gridding are processed in
# parallelized chunks in this script. To disable the chunking, 
# remove the -ind_s and -ind_e input arguments from the relevant
# python script calls.
#
rootdir="$1" # 1st input arg
ncores="$2" # 2nd input arg
#
# Activate conda environment for https://github.com/mikapm/wass-pyfuns
eval "$(conda shell.bash hook)"
conda activate wass
#
# Loop over requested datetimes
for datetime in "${@:3}"; do
    #
    # Set inputs #
    d=${datetime:0:8} # date string
    t=${datetime:8:13} # time string
    expdir_base="$rootdir"/"$d" # Experiment date directory
    expdir="$expdir_base"/"$t" # Experiment time directory
    echo " "
    echo "date: ""$d"
    echo "start time: ""$t"
    echo "Experiment directory: ""$expdir"
    #
    # Check if expdir_base exists
    if [ ! -d "$expdir_base" ]; then
        echo "$expdir_base"" does not exist, generating it ..."
        mkdir "$expdir_base"
    fi
    # Check if expdir exists 
    if [ ! -d "$expdir" ]; then
        echo "$expdir"" does not exist, generating it ..."
        mkdir "$expdir"
    fi
    #
    # Copy over config folder (assuming it exists in $rootdir)
    configdir="$expdir"/config
    # Check if config dir. exists 
    if [ ! -d "$configdir" ]; then
        echo "Copying over config folder to ""$configdir"
        cp -r "$rootdir"/config "$expdir"/config
    fi
    #
    # Check if input image dir. (for WASS) exists
    inputdir="$expdir"/input
    if [ ! -d "$inputdir" ]; then
        echo "$inputdir"" does not exist, generating it ..."
        mkdir "$inputdir"
    fi
    # Check if mean plane dir. exists
    planedir="$expdir"/planes
    if [ ! -d "$planedir" ]; then
        echo "$planedir"" does not exist, generating it ..."
        mkdir "$planedir"
    fi
    # Check if grid dir. exists
    griddir="$expdir"/grid
    if [ ! -d "$griddir" ]; then
        echo "$griddir"" does not exist, generating it ..."
        mkdir "$griddir"
    fi
    # Run file conversion raw -> tif (assuming raw frames in expdir/raw folder)
    echo " "
    echo "Converting frames ..."
    # python /home/mikapm/Github/wass-pyfuns/pywass/prep_images.py "$expdir"/raw -outdir $inputdir
    #
    # Run the WASS processing #
    echo " "
    echo "Running WASS ..."
    python /home/mikapm/Github/wass-pyfuns/pywass/wass_launch.py -dr "$expdir" -pc "$ncores" 
    #
    # Calculate mean planes. Use batches to run in parallel #
    echo " "
    echo "Computing mean planes ..."
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 0 -ind_e 100 &
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 101 -ind_e 200 &
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 201 -ind_e 300 &
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 301 -ind_e 400 &
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 401 -ind_e 500 &
    wait # This will wait until all above scripts finish
    # Average batched planes
    python /home/mikapm/Github/wass-pyfuns/pywass/mean_plane.py -dr "$expdir" -ind_s 401 -ind_e 500 --avg_planes
    #
    # Run the gridding. Use batches to run in parallel #
    echo " "
    echo "Running gridding ..."
    python /home/mikapm/Github/wass-pyfuns/pywass/mesh_to_ncgrid_v2_ekok.py -dr "$rootdir" -date "$datetime" -dxy 0.5 -ind_s 0 -ind_e 100 -step 5 --imgrid &
    python /home/mikapm/Github/wass-pyfuns/pywass/mesh_to_ncgrid_v2_ekok.py -dr "$rootdir" -date "$datetime" -dxy 0.5 -ind_s 101 -ind_e 200 -step 5 --imgrid &
    python /home/mikapm/Github/wass-pyfuns/pywass/mesh_to_ncgrid_v2_ekok.py -dr "$rootdir" -date "$datetime" -dxy 0.5 -ind_s 201 -ind_e 300 -step 5 --imgrid &
    python /home/mikapm/Github/wass-pyfuns/pywass/mesh_to_ncgrid_v2_ekok.py -dr "$rootdir" -date "$datetime" -dxy 0.5 -ind_s 301 -ind_e 400 -step 5 --imgrid &
    python /home/mikapm/Github/wass-pyfuns/pywass/mesh_to_ncgrid_v2_ekok.py -dr "$rootdir" -date "$datetime" -dxy 0.5 -ind_s 401 -ind_e 500 -step 5 --imgrid &
    wait # This will wait until all above scripts finish
    #
    # Delete the input directory that was created, which contains a tif version of every image for both cams #
    # rm -r "${expdir}"/input
    #
    # Make a softlink to the nc file in the dist directory #
    # Note: if no /grid/wass.nc file was made, the datetime is skipped, won't show up in dist
    # if test -f "$expdir"/grid/wass.nc; then
    #     distdir="path/stereo/dist"
    #     ln -rs "$expdir"/grid/wass.nc "$distdir"/"$d"_"$t"_wass.nc 
    # fi
    #
    echo " "
    echo "Processing done for ""$expdir"
done
