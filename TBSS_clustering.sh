#!/bin/bash

### Select directory with TBSS files
### Create mask of significantly different voxels t > 2.5 and of cluster size > 15
cluster -i /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v1/stats/tbss_tstat1.nii.gz -t 2.5 --minextent=15 --oindex=/Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_v1.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_v1.nii.gz -bin /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v1.nii.gz
cluster -i /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/tbss_tstat1.nii.gz -t 2.5 --minextent=15 --oindex=/Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_v2.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_v2.nii.gz -bin /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v2.nii.gz
cluster -i /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/tbss_tstat1.nii.gz -t 2.5 --minextent=15 --oindex=/Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_v3.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_v3.nii.gz -bin /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v3.nii.gz
cluster -i /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v1/stats/tbss_tstat2.nii.gz -t 2.5 --minextent=15 --oindex=/Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_v1.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_v1.nii.gz -bin /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v1.nii.gz
cluster -i /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/tbss_tstat2.nii.gz -t 2.5 --minextent=15 --oindex=/Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_v2.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_v2.nii.gz -bin /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v2.nii.gz
cluster -i /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/tbss_tstat2.nii.gz -t 2.5 --minextent=15 --oindex=/Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_v3.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_v3.nii.gz -bin /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v3.nii.gz
### To compare variability between visits, we must filter out differences between skeleton masks
# fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v1/stats/mean_FA_skeleton_mask.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/mean_FA_skeleton_mask.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/mean_FA_skeleton_mask.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/skeleton_mask_all.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v1.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/skeleton_mask_all.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v1.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v2.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/skeleton_mask_all.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v2.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v3.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/skeleton_mask_all.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v3.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v1.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/skeleton_mask_all.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v1.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v2.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/skeleton_mask_all.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v2.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v3.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/skeleton_mask_all.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v3.nii.gz
### Apply masks to FA values
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v1/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v1.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat1_v1.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v1.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat1_v2.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v1.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat1_v3.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v1/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v1.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat2_v1.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v1.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat2_v2.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v1.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat2_v3.nii.gz

fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v2.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v2/FA_tstat1_v2.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat1_mask_v3.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v3/FA_tstat1_v3.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v2.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v2/FA_tstat2_v2.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/thr2.5_clus15_tstat2_mask_v3.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v3/FA_tstat2_v3.nii.gz

fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v1/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/conjunction_tstat1_mask.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat1_v1_conjunction.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v1/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/conjunction_tstat2_mask.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat2_v1_conjunction.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/conjunction_tstat1_mask.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v2/FA_tstat1_v2_conjunction.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v2/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/conjunction_tstat2_mask.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v2/FA_tstat2_v2_conjunction.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/conjunction_tstat1_mask.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v3/FA_tstat1_v3_conjunction.nii.gz
fslmaths /Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v3/stats/all_FA_skeletonised.nii.gz -mul /Volumes/PT_DATA2/Marc-Antoine/TBSS/mask/FA/conjunction_tstat2_mask.nii.gz /Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v3/FA_tstat2_v3_conjunction.nii.gz