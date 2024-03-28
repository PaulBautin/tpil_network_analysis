from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse du squelette TBSS
#
# example: python connectivity_analysis.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Marc Antoine
#
# Prerequis: environnement virtuel avec python, pandas, numpy, netneurotools, scilpy et matplotlib (env_tpil)
#
#########################################################################################


# Parser
#########################################################################################

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

def tbss_extractor2(visit=1, contrast=1, conjunction=False):

    # Load the 4D image containing FA values for all subjects
    if conjunction == True:
        img = nib.load("/Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v" + str(visit) + "/FA_tstat" + str(contrast) + "_v" + str(visit) + "_conjunction.nii.gz")
    else:
        img = nib.load("/Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/v1/FA_tstat" + str(contrast) + "_v" + str(visit) + ".nii.gz")

    # Get the data array
    fa_data = img.get_fdata()

    # Reshape the data array to have subjects in the first dimension
    fa_data_reshaped = fa_data.reshape((-1, fa_data.shape[-1]))

    # Create a DataFrame to store FA values for each subject
    df = pd.DataFrame(fa_data_reshaped, columns=[f"Subject_{i}" for i in range(fa_data.shape[-1])])
    filtered_df = df.loc[(df != 0).any(axis=1)]

    if visit == 1:
        filtered_df.drop(columns=['Subject_2', 'Subject_6', 'Subject_16','Subject_18'], inplace=True)
        df_clbp = filtered_df.iloc[:, :23]
        df_con = filtered_df.iloc[:, 23:]
        if conjunction == True:
            df_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/clbp/FA_tstat' + str(contrast) + '_v' + str(visit) + '_conjunction.csv')
            df_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/con/FA_tstat' + str(contrast) + '_v' + str(visit) + '_conjunction.csv')
        else:
            df_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/clbp/FA_tstat' + str(contrast) + '_v' + str(visit) + '.csv')
            df_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/con/FA_tstat' + str(contrast) + '_v' + str(visit) + '.csv')
    elif visit == 2:
        filtered_df.drop(columns=['Subject_5', 'Subject_25'], inplace=True)
        df_clbp = filtered_df.iloc[:, :23]
        df_con = filtered_df.iloc[:, 23:]
        if conjunction == True:
            df_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/clbp/FA_tstat' + str(contrast) + '_v' + str(visit) + '_conjunction.csv')
            df_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/con/FA_tstat' + str(contrast) + '_v' + str(visit) + '_conjunction.csv')
        else:
            df_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/clbp/FA_tstat' + str(contrast) + '_v' + str(visit) + '.csv')
            df_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/con/FA_tstat' + str(contrast) + '_v' + str(visit) + '.csv')
    elif visit == 3:
        filtered_df.drop(columns=['Subject_24'], inplace=True)
        df_clbp = filtered_df.iloc[:, :23]
        df_con = filtered_df.iloc[:, 23:]
        if conjunction == True:
            df_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/clbp/FA_tstat' + str(contrast) + '_v' + str(visit) + '_conjunction.csv')
            df_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/con/FA_tstat' + str(contrast) + '_v' + str(visit) + '_conjunction.csv')
        else:
            df_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/clbp/FA_tstat' + str(contrast) + '_v' + str(visit) + '.csv')
            df_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/con/FA_tstat' + str(contrast) + '_v' + str(visit) + '.csv')
    else:
        raise ValueError("Invalid visit specified. Allowed visits: 1, 2 or 3")

    return df_clbp, df_con

def friedman(tbss_v1, tbss_v2, tbss_v3, condition='clbp', contrast=1, conjunction=False):
    """
    Calculates friedman test (non-parametric repeated measures ANOVA) to measure variability
    of tbss results at different time points

    Parameters
    ----------
    df_centrality_v1 : (S, N) pandas DataFrame
        DataFrame containing TBSS values for every subject S and every voxel N for version 1.
    df_centrality_v2 : (S, N) pandas DataFrame
        DataFrame containing TBSS values for every subject S and every voxel N for version 2.
    df_centrality_v3 : (S, N) pandas DataFrame
        DataFrame containing TBSS values for every subject S and every voxel N for version 3.

    Returns
    -------
    df_results : pandas DataFrame
        DataFrame of chi-square statistic and p-value for each ROI.
    """
    # Create an empty DataFrame to store Friedman test results for each voxel
    chi_stats = pd.DataFrame(index=tbss_v1.index, columns=['Friedman_Stat'])
    p_values = pd.DataFrame(index=tbss_v1.index, columns=['Friedman_P_Value'])

    # Iterate over each row (voxel) in the DataFrames
    for index, row in tbss_v1.iterrows():
        # Get the FA values for this voxel across all visits
        fa_values_visit1 = row.values
        fa_values_visit2 = tbss_v2.loc[index].values
        fa_values_visit3 = tbss_v3.loc[index].values
        
        # Perform the Friedman test for this voxel
        chi_stat, p_value = stats.friedmanchisquare(fa_values_visit1, fa_values_visit2, fa_values_visit3)
        
        # Store the chi-sqaure stat and p-value in the DataFrame
        p_values.at[index, 'Friedman_P_Value'] = p_value
        chi_stats.at[index, 'Friedman_Stat'] = chi_stat

    if conjunction == True:
        chi_stats.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/' + str(condition) + '/friedman' + str(contrast) + 'fa_conjunction.csv', index=True)
    else:
        chi_stats.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/results_tbss/FA/' + str(condition) + '/friedman' + str(contrast) + 'fa.csv', index=True)

    return chi_stats, p_values

def associate_friedman_statistic_with_voxels(chi_stats_df, original_img):
    """
    Associates Friedman test statistic values with corresponding voxels in the original 4D volume.

    Parameters:
        - chi_stats_df: pandas DataFrame containing the Friedman test statistics for each voxel.
        - original_img: nibabel Nifti1Image object representing the original 4D volume.

    Returns:
        - friedman_img: nibabel Nifti1Image object representing the 3D volume with Friedman test statistics.
    """
    # Get the data array from the original image
    original_data = original_img.get_fdata()

    # Create an empty array to store Friedman test statistic values
    friedman_data = np.zeros(original_data.shape[:-1])

    # Iterate over each voxel
    for index, row in chi_stats_df.iterrows():
        # Get the voxel index
        voxel_index = np.unravel_index(index, friedman_data.shape)
        
        # Get the Friedman test statistic value for this voxel
        friedman_statistic = row['Friedman_Stat']
        
        # Assign the Friedman test statistic value to the corresponding voxel
        friedman_data[voxel_index] = friedman_statistic

    # Create a new nibabel image with the Friedman test statistic values
    friedman_img = nib.Nifti1Image(friedman_data, original_img.affine)

    return friedman_img

def friedman_variability_img(condition='clbp', contrast=1, conjunction=False):
    # Extract data from FA valus from image
    fa_values_v1 = tbss_extractor2(visit=1, contrast=contrast, conjunction=conjunction)
    fa_values_v2 = tbss_extractor2(visit=2, contrast=contrast, conjunction=conjunction)
    fa_values_v3 = tbss_extractor2(visit=3, contrast=contrast, conjunction=conjunction)
    # Run Friedman test
    friedman_test = friedman(fa_values_v1,fa_values_v2,fa_values_v3,condition=condition,contrast=contrast,conjunction=conjunction)
    # Load original image
    original_img = nib.load("/Volumes/PT_DATA2/Marc-Antoine/TBSS/data/v1/stats/all_FA_skeletonised.nii.gz")
    friedman_img = associate_friedman_statistic_with_voxels(friedman_test, original_img)
    nib.save(friedman_img, "/Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/friedman/" + str(condition) + str(contrast) + "_fa.nii.gz")

    if conjunction == True:
        nib.save(friedman_img, "/Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/friedman/" + str(condition) + str(contrast) + "_fa_conjunction.nii.gz")
    else:
        nib.save(friedman_img, "/Volumes/PT_DATA2/Marc-Antoine/TBSS/results/clustered/friedman/" + str(condition) + str(contrast) + "_fa.nii.gz")


def main():

    friedman_variability_img('clbp', contrast=1, conjunction=False)
    friedman_variability_img('clbp', contrast=2, conjunction=False)
    friedman_variability_img('con', contrast=1, conjunction=False)
    friedman_variability_img('con', contrast=2, conjunction=False)
    friedman_variability_img('clbp', contrast=1, conjunction=True)
    friedman_variability_img('clbp', contrast=2, conjunction=True)
    friedman_variability_img('con', contrast=1, conjunction=True)
    friedman_variability_img('con', contrast=2, conjunction=True)

if __name__ == "__main__":
    main()