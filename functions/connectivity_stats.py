from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse de la connectivite avec commmit2
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

import pandas as pd
import numpy as np
import os
import argparse
import bct
import pingouin as pg
import matplotlib
from functions.connectivity_read_files import find_files_with_common_name
from scipy.stats import ttest_rel
from scipy.stats import t
from scipy.stats import zscore
from scipy import stats

def get_parser():
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Compute statistics based on the .csv files containing the tractometry metrics:",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-clbp",
        required=True,
        default='connectivity_results',
        help='Path to folder that contains output .csv files (e.g. "~/dev_tpil/tpil_network_analysis/data/22-11-16_connectoflow/clbp/sub-pl007_ses-v1/Compute_Connectivity")',
    )
    mandatory.add_argument(
        "-con",
        required=True,
        default='connectivity_results',
        help='Path to folder that contains output .csv files (e.g. "~/dev_tpil/tpil_network_analysis/data/22-11-16_connectoflow/control/sub-pl029_ses-v1/Compute_Connectivity")',
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-fig',
        help='Generate figures',
        action='store_true'
    )
    optional.add_argument(
        '-o',
        help='Path where figures will be saved. By default, they will be saved in the current directory.',
        default="."
    )
    return parser

def mean_matrix(df_connectivity_matrix):
    """
    Returns mean value of all subjects for a given edge

    Parameters
    ----------
    df_connectivity_matrix : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    
    Returns
    -------
    mean_matrix : (N, N) pandas DataFrame
    """
    mean_matrix = df_connectivity_matrix.groupby("roi").mean()  
    return mean_matrix

def z_score(df_con_v1, df_clbp_v1):
    """
    Returns mean z-score between clbp and con data for a given edge

    Parameters
    ----------
    df_con_v1 : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clbp_v1 : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    df_anor_mean : (N, N) pandas DataFrame
    """
    df_con_mean = mean_matrix(df_con_v1)
    df_con_std = df_con_v1.groupby("roi").std()
    #df_clbp = df_clbp_v1.set_index(df_clbp_v1.index.names) # step necessary to apply z-score equation on every subject
    df_anor = (df_clbp_v1 - df_con_mean) / df_con_std
    df_anor_mean = mean_matrix(df_anor)
    return df_anor_mean

def nbs_data(df_con_v1, df_clbp_v1, save_path):
    """
    Performs Network-based statistics between clbp and control group and save the results in results_nbs
    Zalesky A, Fornito A, Bullmore ET (2010) Network-based statistic: Identifying differences in brain networks. NeuroImage.
    10.1016/j.neuroimage.2010.06.041

    Parameters
    ----------
    df_con_v1 : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clbp_v1 : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    save_path : directory ex: '/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/23-07-11_v1_'

    Returns
    -------
    pval : Cx1 np.ndarray. A vector of corrected p-values for each component of the networksidentified. If at least one p-value 
    is less than alpha, the omnibus null hypothesis can be rejected at alpha significance. The nullhypothesis is that the value 
    of the connectivity from each edge hasequal mean across the two populations.
    adj : IxIxC np.ndarray. An adjacency matrix identifying the edges comprising each component.edges are assigned indexed values.
    null : Kx1 np.ndarray. A vector of K sampled from the null distribution of maximal component size.
    """
    # transform to 3d numpy array (N, N, S) with N nodes and S subjects
    np_con_v1 = np.dstack(list(df_con_v1.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    np_clbp_v1 = np.dstack(list(df_clbp_v1.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    pval, adj, null = bct.nbs.nbs_bct(np_con_v1, np_clbp_v1, thresh=2.0, tail='both', paired=False, verbose=True)
    np.save(save_path + 'pval.npy', pval)
    np.save(save_path + 'adj.npy', adj)
    np.save(save_path + 'null.npy', null)
    return pval, adj, null

def paired_t_test(df_v1, df_v2):
    """
    Calculates t-test of paired data. For example, calculates paired t-test of degree centrality at v1 versus v2

    Parameters
    ----------
    df_v1 : (1, N) pandas DataFrame where N is the number of nodes 
    df_v2 : (1, N) pandas DataFrame where N is the number of nodes

    Returns
    -------
    df_t_test_results : (2, N) pandas dataframe with t statistic and p-value where N is the number of nodes
    prints float: Critical t-value.
    t_test_results.txt : txt file with t statistic for every node
    """
    ### Calculate paired t-tests for each ROI
    merged_df = pd.merge(df_v1, df_v2, on=['subject', 'roi'], suffixes=('_v1', '_v2'))
    results = {}
    for roi, data in merged_df.groupby('roi'):
        t_statistic, p_value = ttest_rel(data['centrality_v1'], data['centrality_v2'])
        results[roi] = {'t_statistic': t_statistic, 'p_value': p_value}
    ### Create a new DataFrame to store the t-test results
    df_t_test_results = pd.DataFrame(results).T
    alpha = 0.05 
    degrees_of_freedom = df_v1.index.get_level_values('subject').nunique() - 1
    critical_t = t.ppf(1 - alpha / 2, degrees_of_freedom)
    print(f"Critical t-value for {alpha}-level significance: {critical_t}")
    np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/t_test_results.txt', df_t_test_results['t_statistic'], fmt='%1.5f')
    return df_t_test_results

def friedman(df_centrality_v1, df_centrality_v2, df_centrality_v3):
    """
    Calculates friedman test (non-parametric repeated measures ANOVA) to measure variability
    of centrality metrics at different time points

    Parameters
    ----------
    df_centrality_v1 : pandas DataFrame
        DataFrame containing centrality data for version 1.
    df_centrality_v2 : pandas DataFrame
        DataFrame containing centrality data for version 2.
    df_centrality_v3 : pandas DataFrame
        DataFrame containing centrality data for version 3.

    Returns
    -------
    df_results : pandas DataFrame
        DataFrame of chi-square statistic and p-value for each ROI.
    """
    unique_rois = df_centrality_v1['roi'].unique()
    results_statistics = []
    results_pval = []

    for roi in unique_rois:
        roi_data_v1 = df_centrality_v1[df_centrality_v1['roi'] == roi]['centrality'].values
        roi_data_v2 = df_centrality_v2[df_centrality_v2['roi'] == roi]['centrality'].values
        roi_data_v3 = df_centrality_v3[df_centrality_v3['roi'] == roi]['centrality'].values

        result = stats.friedmanchisquare(roi_data_v1, roi_data_v2, roi_data_v3)
        results_statistics.append({'roi': roi, 'statistic': result.statistic})
        results_pval.append({'roi': roi, 'p_value': result.pvalue})

    # Create DataFrames to store the results
    df_results_stats = pd.DataFrame(results_statistics)
    df_results_pval = pd.DataFrame(results_pval)

    return df_results_stats, df_results_pval
    
def icc(df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3):
    """
    Calculates intraclass correlation coefficient
    of centrality metrics at different time points

    Parameters
    ----------
    df_clean_centrality_v1 : (1, N) pandas DataFrame where N is the number of nodes 
    df_clean_centrality_v2 : (1, N) pandas DataFrame where N is the number of nodes
    df_clean_centrality_v3 : (1, N) pandas DataFrame where N is the number of nodes

    Returns
    -------
    icc : list of all types of ICC with their description, value, F-value, degrees of freedom, p-value and CI95%
    """
    df_clean_centrality_v1.insert(1, "session", 1, True)
    df_clean_centrality_v2.insert(1, "session", 2, True)
    df_clean_centrality_v3.insert(1, "session", 3, True)
    df_concat = pd.concat([df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3])
    df_concat_reset = df_concat.reset_index()
    df_concat_reset['centrality'] = df_concat_reset['centrality']
    print(df_concat)
    print(df_concat_reset)
    # print(df_concat_reset.groupby(['roi','session']).std())
    # print(df_concat_reset.groupby(['subject', 'roi']).std())
    icc = pg.intraclass_corr(data=df_concat_reset, targets='roi', raters='session', ratings='centrality')
    icc.set_index('Type')
    print(icc)
    return icc
    
def my_icc(df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3):
    """
    Calculates intraclass correlation coefficient
    of centrality metrics at different time points

    Parameters
    ----------
    df_clean_centrality_v1 : (1, N) pandas DataFrame where N is the number of nodes 
    df_clean_centrality_v2 : (1, N) pandas DataFrame where N is the number of nodes
    df_clean_centrality_v3 : (1, N) pandas DataFrame where N is the number of nodes

    Returns
    -------
    icc : list of all types of ICC with their description, value, F-value, degrees of freedom, p-value and CI95%
    """
    # Specify session in DataFrame
    df_clean_centrality_v1.insert(1, "session", 1, True)
    df_clean_centrality_v2.insert(1, "session", 2, True)
    df_clean_centrality_v3.insert(1, "session", 3, True)
    # Merge DataFrames togetther and reorder index
    df_concat = pd.concat([df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3])
    df_concat_reset = df_concat.reset_index()
    df_concat_reset['centrality'] = df_concat_reset['centrality']
    # Within-subject Variance: variance of time points for each subject in each region
    within_sub_var = df_concat_reset.groupby(['subject', 'roi']).var()
    print(within_sub_var)
    within_sub_var_region = mean_matrix(within_sub_var)
    print("within-subject variance per region:", within_sub_var_region)
    within_sub_var_mean = within_sub_var_region['centrality'].mean()
    print("within-subject/region variance:", within_sub_var_mean)
    # Between-subject Variance: variance of subjects
    mean_session_centrality = df_concat_reset.groupby(['subject', 'roi'])['centrality'].mean().reset_index()
    between_sub_var_region = mean_session_centrality.groupby('roi').var()
    print("between-subject variance per region:", between_sub_var_region)
    between_sub_var_mean = mean_session_centrality.var()
    print("between-subject/region variance:", between_sub_var_mean)
    # Calculate the ICC
    icc = (between_sub_var_mean - within_sub_var_mean) / (between_sub_var_mean + (3-1) * within_sub_var_mean)
    print("intraclass correlation coefficient:", icc)
    icc_region = (between_sub_var_region - within_sub_var_region) / (between_sub_var_region + (3-1) * within_sub_var_region)
    print("intraclass correlation coefficient per region:", icc_region)
    print("mean icc_region:", icc_region['centrality'].mean())

def main():
    """
    main function, gather stats and call plots
    """
    ### Get parser elements
    parser = get_parser()
    arguments = parser.parse_args()
    path_results_con = os.path.abspath(os.path.expanduser(arguments.con))
    path_results_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    path_output = os.path.abspath(arguments.o)

    df_con = find_files_with_common_name(path_results_con, "commit2_weights.csv")
    df_clbp = find_files_with_common_name(path_results_clbp, "commit2_weights.csv")

    df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)
    df_con_v2 = df_con[df_con['session'] == "v2"].drop("session", axis=1)
    df_clbp_v2 = df_clbp[df_clbp['session'] == "v2"].drop("session", axis=1)
    df_con_v3 = df_con[df_con['session'] == "v3"].drop("session", axis=1)
    df_clbp_v3 = df_clbp[df_clbp['session'] == "v3"].drop("session", axis=1)


if __name__ == "__main__":
    main()