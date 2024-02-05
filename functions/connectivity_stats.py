from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Fonctions pour l'analyse statistique de la connectivité structurelle
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
import bct
import pingouin as pg
from functions.connectivity_read_files import find_files_with_common_name
from scipy.stats import ttest_rel, t
from scipy import stats

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

def difference(df_con, df_clbp):
    """
    Returns mean difference between clbp and con data for a given roi

    Parameters
    ----------
    df_con_v1 : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clbp_v1 : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    df_diff : (N, N) pandas DataFrame
    """
    df_diff = (df_clbp - df_con)
    
    return df_diff

def nbs_data(df_con_v1, df_clbp_v1, thr, save_path):
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
    np_con_v1 = np.dstack(list(df_con_v1.groupby(['subject']).apply(lambda x: x.to_numpy())))
    np_clbp_v1 = np.dstack(list(df_clbp_v1.groupby(['subject']).apply(lambda x: x.to_numpy())))
    pval, adj, null = bct.nbs.nbs_bct(np_con_v1, np_clbp_v1, thresh=thr, tail='both', paired=False, verbose=True)
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
    df_centrality_v1 : (1, SxN) pandas DataFrame
        DataFrame containing centrality data for version 1.
    df_centrality_v2 : (1, SxN) pandas DataFrame
        DataFrame containing centrality data for version 2.
    df_centrality_v3 : (1, SxN) pandas DataFrame
        DataFrame containing centrality data for version 3.

    Returns
    -------
    df_results : pandas DataFrame
        DataFrame of chi-square statistic and p-value for each ROI.
    """
    unique_rois = df_centrality_v1.index.get_level_values('roi').unique()
    results_statistics = []
    results_pval = []

    for roi in unique_rois:
        roi_data_v1 = df_centrality_v1.loc[df_centrality_v1.index.get_level_values('roi') == roi, 'centrality'].values
        roi_data_v2 = df_centrality_v2.loc[df_centrality_v2.index.get_level_values('roi') == roi, 'centrality'].values
        roi_data_v3 = df_centrality_v3.loc[df_centrality_v3.index.get_level_values('roi') == roi, 'centrality'].values

        result = stats.friedmanchisquare(roi_data_v1, roi_data_v2, roi_data_v3)
        results_statistics.append({'roi': roi, 'statistic': result.statistic})
        results_pval.append({'roi': roi, 'p_value': result.pvalue})

    # Create DataFrames to store the results
    df_results_stats = pd.DataFrame(results_statistics)
    df_results_stats.set_index('roi', inplace=True)
    df_results_pval = pd.DataFrame(results_pval)
    df_results_pval.set_index('roi', inplace=True)

    return df_results_stats, df_results_pval
    
def icc(df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3):
    """
    Calculates intraclass correlation coefficient
    of centrality metrics at different time points

    Parameters
    ----------
    df_clean_centrality_v1 : (1, NxS) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clean_centrality_v2 : (1, NxS) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clean_centrality_v3 : (1, NxS) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    icc : list of all types of ICC with their description, value, F-value, degrees of freedom, p-value and CI95%
    """
    df_clean_centrality_v1.insert(1, "session", 1, True)
    df_clean_centrality_v2.insert(1, "session", 2, True)
    df_clean_centrality_v3.insert(1, "session", 3, True)
    df_concat = pd.concat([df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3])
    within, between = calculate_cv(df_concat)
    print(within)
    print(between)
    df_concat_reset = df_concat.reset_index()
    icc = pg.intraclass_corr(data=df_concat_reset, targets='subject', raters='session', ratings='metric')
    icc.set_index('Type')
    print("intraclass correlation coefficient:", icc)
    
    return icc

def calculate_cv(dataframe):
    # Calculate mean and standard deviation within each subject and session
    mean_values_within_subject = dataframe.groupby(['subject'])['metric'].mean()
    std_values_within_subject = dataframe.groupby(['subject'])['metric'].std()
    
    # Calculate coefficient of variability within each subject and session
    cv_within_subject_session = (std_values_within_subject / mean_values_within_subject) * 100

    # Calculate mean and standard deviation across all subjects and sessions
    mean_values_between_subject = dataframe.groupby(['session'])['metric'].mean()
    std_values_between_subject = dataframe.groupby(['session'])['metric'].std()
    
    # Calculate coefficient of variability across all subjects and sessions
    cv_between_subject = (std_values_between_subject / mean_values_between_subject) * 100

    return cv_within_subject_session, cv_between_subject

def my_icc(df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3):
    """
    Calculates intraclass correlation coefficient
    of centrality metrics at different time points

    Parameters
    ----------
    df_clean_centrality_v1 : (1, SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clean_centrality_v2 : (1, SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clean_centrality_v3 : (1, SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    icc : list of all types of ICC with their description, value, F-value, degrees of freedom, p-value and CI95%
    """
    # Specify session in DataFrame
    df_clean_centrality_v1.insert(1, "session", 1, True)
    n1 = len(df_clean_centrality_v1['metric'])
    df_clean_centrality_v2.insert(1, "session", 2, True)
    n2 = len(df_clean_centrality_v2['metric'])
    df_clean_centrality_v3.insert(1, "session", 3, True)
    n3 = len(df_clean_centrality_v3['metric'])
    
    
    # Merge DataFrames togetther and reorder index
    df_concat = pd.concat([df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3])

    # # Calculate each sum of squared differences for every session
    # ssd1 = ((df_clean_centrality_v1['metric'] - df_clean_centrality_v1['metric'].mean())**2).sum()
    # ssd2 = ((df_clean_centrality_v2['metric'] - df_clean_centrality_v2['metric'].mean())**2).sum()
    # ssd3 = ((df_clean_centrality_v3['metric'] - df_clean_centrality_v3['metric'].mean())**2).sum()
    # ssd_list = [ssd1, ssd2, ssd3]
    # print(ssd1)
    # # Calculate sum of squares within the groups
    # ssdw = sum(ssd_list)
    # print(ssdw)
    # # Calulate the sum of squares between the groups
    # ssdb1 = n1*(df_clean_centrality_v1['metric'].mean() - df_concat['metric'].mean())**2
    # ssdb2 = n2*(df_clean_centrality_v2['metric'].mean() - df_concat['metric'].mean())**2
    # ssdb3 = n3*(df_clean_centrality_v3['metric'].mean() - df_concat['metric'].mean())**2
    # ssdb_list = [ssdb1, ssdb2, ssdb3]
    # print(ssdb1)
    # ssdb = sum(ssdb_list)
    # print(ssdb)
    # # Calculate mean square within the groups
    # msw = ssdw / (n1 - 3)
    # # Calculate the mean square between the groups
    # msb = ssdb / (3 - 1)

    # # Calculate the icc
    # icc = (msb - msw) / (msb + (3-1) * msw)

    # Within-subject Variance: variance of time points for each subject in each region
    within_sub_var = df_concat.groupby(['subject', 'roi'])['metric'].var()
    print("Variability accross session for every roi in every subject", within_sub_var)
    within_sub_var_mean = within_sub_var.mean()
    print("within-subject variance:", within_sub_var_mean)
    # Between-subject Variance: variance of subjects
    between_sub_var = df_concat.groupby(['roi', 'session'])['metric'].var()
    print("Variability accross subjects for every session in every roi", between_sub_var)
    between_sub_var_mean = between_sub_var.mean()
    print("between-subject variance:", between_sub_var_mean)
    
    # Calculate the ICC
    icc = (between_sub_var_mean - within_sub_var_mean) / (between_sub_var_mean + (3-1) * within_sub_var_mean)
    print("intraclass correlation coefficient:", icc)

    return icc

def calculate_icc_all_rois(dataframe, metric):
    # Filter the dataframe for the given metric
    subset = dataframe.loc[:, ['subject', 'visit', metric]]

    # Create a long format for the intraclass_corr function
    long_format = subset.reset_index()
    
    # Calculate ICC
    result = pg.intraclass_corr(data=long_format, targets='subject', raters='visit', ratings=metric, nan_policy='omit')

    print(f"Intraclass correlation coefficient for {metric}:\n{result}")
    return result
