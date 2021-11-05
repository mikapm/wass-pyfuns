#!/bin/bash
for starttime in $*; do
       python /lustre/storeB/users/mikapm/stereo-wave/sw_pyfuns/wass_stuff/wass_launch.py -dr /lustre/storeB/project/fou/om/stereowave/data/wass_output/20200521/$starttime -pc 24
done
