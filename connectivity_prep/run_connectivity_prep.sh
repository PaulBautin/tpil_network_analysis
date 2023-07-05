#!/bin/bash

# This would run the TPIL Bundle Segmentation Pipeline with the following resources:
#     Prebuild Singularity images: https://scil.usherbrooke.ca/pages/containers/
#     Brainnetome atlas in MNI space: https://atlas.brainnetome.org/download.html
#     FA template in MNI space: https://brain.labsolver.org/hcp_template.html


#my_singularity_img='/home/pabaua/dev_scil/containers/containers_scilus_1.5.0.sif' # or .sif
my_singularity_img='/home/pabaua/neurodocker_container/singularity_container.sif' # or .sif
my_main_nf='/home/pabaua/dev_tpil/tpil_network_analysis/connectivity_prep/main.nf'


my_input_tr='/home/pabaua/dev_tpil/results/results_tracto'
my_input_fs='/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output'
my_template='/home/pabaua/dev_scil/atlas/mni_masked.nii.gz'
my_data_fs='/home/pabaua/dev_tpil/tpil_network_analysis/connectivity_prep'


nextflow run $my_main_nf  \
  --input_tr $my_input_tr \
  --input_fs $my_input_fs \
  --template $my_template \
  -with-singularity $my_singularity_img \
  -resume

################################
#singularity {
#    enabled = true
#    runOptions = "--cleanenv -B $SINGULARITY_BIND_BUNDLE"
#    envWhitelist = ['SUBJECTS_DIR','ANTSPATH']
#    autoMounts = true
#}
#
#process {
#    withName:Create_sub_mask {
#        container = '/home/pabaua/neurodocker_container/new_bundle_container.sif'
#    }
#    withName:freesurfer_to_subject {
#        container = '/home/pabaua/neurodocker_container/new_bundle_container.sif'
#    }
#}

#    beforeScript "export FS_LICENSE=/home/pabaua/dev_tpil/data/Freesurfer/license.txt"
#    beforeScript "export FREESURFER_HOME=/usr/local/freesurfer/7.3.2"
#    beforeScript "export SUBJECTS_DIR=/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output"
#    beforeScript "source /usr/local/freesurfer/7.3.2/SetUpFreeSurfer.sh"
################################