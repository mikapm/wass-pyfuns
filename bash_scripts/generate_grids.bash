#!/bin/bash
for starttime in $@; do
       python /lustre/storeB/users/mikapm/stereo-wave/sw_pyfuns/wass_stuff/mesh_to_ncgrid.py -dr /lustre/storeB/project/fou/om/stereowave/data/wass_output/20200521/$starttime
done
