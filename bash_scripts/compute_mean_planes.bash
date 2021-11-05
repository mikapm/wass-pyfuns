#!/bin/bash
for starttime in $*; do
       python /lustre/storeB/users/mikapm/stereo-wave/sw_pyfuns/wass_stuff/mean_plane.py -dr /lustre/storeB/project/fou/om/stereowave/data/wass_output/20200521/$starttime &
done
