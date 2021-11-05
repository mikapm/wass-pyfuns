#!/bin/bash
for starttime in $*; do
   echo $starttime
   python /lustre/storeB/users/mikapm/stereo-wave/sw_pyfuns/wass_stuff/prep_images.py /lustre/storeB/project/fou/om/stereowave/data/20200521_extracted/$starttime &
done
