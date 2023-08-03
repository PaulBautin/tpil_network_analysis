#!/bin/bash

#select directory with TBSS files
cd /mnt/d/Marc-Antoine/TBSS/v*

tbss_1_preproc *.nii.gz

tbss_2_reg -T

tbss_3_postreg -S

tbss_4_prestats 0.2

########################################################################

# Create design matrix: design.txt
# Check order with cd FA; imglob *_FA.*
# Ex: 7 patients, 8 controls
# 1 0
# 1 0
# 1 0
# 1 0
# 1 0
# 1 0
# 1 0
# 0 1
# 0 1
# 0 1
# 0 1
# 0 1
# 0 1
# 0 1
# 0 1

# Create design contrast: contrast.txt
# Ex: if Group A = CLBP and Group B = CON
# 1 -1      # C1 A > B
# 1 -1      # C2 B > A
# 1 0      # Mean A
# 0 1      # Mean B
#Text2Vest design.txt design.mat
#Text2Vest contrast.txt design.con

#randomise -i all_FA_skeletonised.nii.gz -o tbss -m mean_FA_skeleton_mask.nii.gz -d design.mat -t design.con -n 500 --T2
# Where:
# -i : input skeletonized FA data for all subjects (4D NIFTI format)
# -o : Output directory where results will be saved
# -d : Design matrix
# -t : Contrast file
# -n : Number of permutations
# --T2 : Threshold-Free Cluster Enhancement option

### tbss_tstat1 = CLBP > CON
### tbss_tstat2 = CON > CLBP

########################################################################
#cd /mnt/d/Marc-Antoine/TBSS/v*/stats
#cluster --in=tbss_tstat1.nii.gz --thresh=2.0 --oindex=/mnt/d/Marc-Antoine/TBSS/cluster_mask/*/cluster1_index_v*.nii.gz
#fslmaths cluster1_index_v*.nii.gz -thr 4 -bin cluster_mask1_min4voxels.nii_v*.gz
#cd /mnt/d/Marc-Antoine/TBSS/cluster_mask/*
#fslmaths cluster1_index_v1.nii.gz -mul cluster1_index_v2.nii.gz -mul cluster1_index_v3.nii.gz cluster_intersection1_mini4voxels.nii.gz