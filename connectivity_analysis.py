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
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
import bct
import networkx as nx
from scipy.stats import ttest_rel
from scipy.stats import t
from statsmodels.stats.anova import AnovaRM
from connectivity_read_files import find_files_with_common_name
from connectivity_filtering import scilpy_filter
from connectivity_filtering import load_brainnetome_centroids
from connectivity_filtering import distance_dependant_filter
from connectivity_filtering import threshold_filter
from connectivity_graphing import circle_graph
from connectivity_graphing import histogram
from connectivity_figures import plot_network
from graph_theory_functions import networkx_degree_centrality
from graph_theory_functions import networkx_betweenness_centrality
from graph_theory_functions import networkx_eigenvector_centrality
from graph_theory_functions import networkx_cluster_coefficient
from graph_theory_functions import z_score_centrality
from graph_theory_functions import networkx_graph_convertor

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

def prepare_data(df_connectivity_matrix, absolute=True):
    """
    Returns Data in the appropriate format for figure functions

    Parameters
    ----------
    df_connectivity_matrix : (N, N) pandas DataFrame

    Returns
    -------
    np_connectivity_matrix : (N, N) array_like
    """
    np_connectivity_matrix = df_connectivity_matrix.values
    np_connectivity_matrix[np.isnan(np_connectivity_matrix)] = 0 
    if absolute:
        np_connectivity_matrix = np.triu(abs(np_connectivity_matrix))
    else:
        np_connectivity_matrix = np.triu(np_connectivity_matrix)
    return np_connectivity_matrix

def data_cleaner(df_connectivity_matrix):
    """
    Returns Data with only the desired patients

    Parameters
    ----------
    df_connectivity_matrix : (NxS rows, 1 column) pandas DataFrame

    Returns
    -------
    df_connectivity_matrix : (NxS-X, N) pandas DataFrame
    """
    subjects_to_remove = ['sub-pl008', 'sub-pl037', 'sub-pl039'] # Need to hard-code this!
    df_cleaned = df_connectivity_matrix[~df_connectivity_matrix.index.get_level_values('subject').isin(subjects_to_remove)]
    return(df_cleaned)

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

    ### Get connectivity data
    df_con = find_files_with_common_name(path_results_con, "commit2_weights.csv")
    df_clbp = find_files_with_common_name(path_results_clbp, "commit2_weights.csv")

    ### work only on one session at a time
    df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)
    df_con_v2 = df_con[df_con['session'] == "v2"].drop("session", axis=1)
    df_clbp_v2 = df_clbp[df_clbp['session'] == "v2"].drop("session", axis=1)
    df_con_v3 = df_con[df_con['session'] == "v3"].drop("session", axis=1)
    df_clbp_v3 = df_clbp[df_clbp['session'] == "v3"].drop("session", axis=1)
    # ### Get similarity data
    # df_con_sc = find_files_with_common_name(path_results_con, "sc.csv")
    # df_clbp_sc = find_files_with_common_name(path_results_clbp, "sc.csv")

    # ### work only on one session at a time
    # df_con_sc_v1 = df_con_sc[df_con['session'] == "v1"].drop("session", axis=1)
    # df_clbp_sc_v1 = df_clbp_sc[df_clbp['session'] == "v1"].drop("session", axis=1)

    """
    To create a Networkx graph of delta degree centrality of Commit2_weights.csv of clbp at v1 and v2 after scilpy filtering
    """
    # ### Scilpy filter on commit2_weights data
    # mask_clbp_commit2_v1 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x))
    # mask_clbp_commit2_v2 = df_clbp_v2.groupby('subject').apply(lambda x:scilpy_filter(x))
    
    # ### Degree centrality
    # df_clbp_centrality_v1 = mask_clbp_commit2_v1.groupby('subject').apply(lambda x:networkx_degree_centrality(x)).rename(columns={0: 'centrality'})
    # df_clbp_centrality_v1.index.names = ['subject', 'roi']
    # df_clean_centrality_v1 = data_cleaner(df_clbp_centrality_v1)
    # df_clbp_centrality_v2 = mask_clbp_commit2_v2.groupby('subject').apply(lambda x:networkx_degree_centrality(x)).rename(columns={0: 'centrality'})
    # df_clbp_centrality_v2.index.names = ['subject', 'roi']
    # df_clean_centrality_v2 = data_cleaner(df_clbp_centrality_v2)
    # ### Calculate paired t-tests for each ROI
    # merged_df = pd.merge(df_clean_centrality_v1, df_clean_centrality_v2, on=['subject', 'roi'], suffixes=('_v1', '_v2'))
    # results = {}
    # for roi, data in merged_df.groupby('roi'):
    #     t_statistic, p_value = ttest_rel(data['centrality_v1'], data['centrality_v2'])
    #     results[roi] = {'t_statistic': t_statistic, 'p_value': p_value}
    # ### Create a new DataFrame to store the t-test results
    # df_t_test_results = pd.DataFrame(results).T
    # print(df_t_test_results)
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/t_test_results.txt', df_t_test_results['t_statistic'], fmt='%1.5f')
    # ### Load numpy array of v1 and v2 and calculate delta
    # delta_v2_v1 = df_clean_centrality_v2 - df_clean_centrality_v1
    
    # ### Mean of mask_clbp_commit2 connectivity matrix for df_connectivity_matrix of networkx_graph_convertor
    # df_clbp_mean_filter = mean_matrix(mask_clbp_commit2_v1)
    # df_clbp_mean_centrality = mean_matrix(delta_v2_v1)
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/delta_v2_v1.txt', df_clbp_mean_centrality, fmt='%1.5f')
    
    # ### Networkx graph of degree centrality of commit2_weights.csv nodes filtered by scilpy
    # networkx_graph_convertor(df_clbp_mean_filter, df_t_test_results)

    """
    To create a Networkx graph of z-score degree centrality of Commit2_weights.csv of clbp and con at v1 after scilpy filtering
    """
    # ### Scilpy filter on commit2_weights data
    # mask_clbp_commit2 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x))
    # mask_con_commit2 = df_con_v1.groupby('subject').apply(lambda x:scilpy_filter(x))
    # ### Degree centrality
    # df_con_centrality = mask_con_commit2.groupby('subject').apply(lambda x:networkx_cluster_coefficient(x)).rename(columns={0: 'centrality'})
    # df_con_centrality.index.names = ['subject', 'roi']
    # df_clbp_centrality = mask_clbp_commit2.groupby('subject').apply(lambda x:networkx_cluster_coefficient(x)).rename(columns={0: 'centrality'})
    # df_clbp_centrality.index.names = ['subject', 'roi']
    # ### Z-score of degree centrality for df_weighted_nodes of networkx_graph_convertor
    # df_z_score_nx = z_score_centrality(df_con_centrality, df_clbp_centrality)
    # print(df_z_score_nx)
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/df_z_score_nx.txt', df_z_score_nx, fmt='%1.3f')
    # ### Mean of mask_clbp_commit2 connectivity matrix for df_connectivity_matrix of networkx_graph_convertor
    # df_clbp_mean_filter = mean_matrix(mask_clbp_commit2)
    # df_clbp_mean_centrality = mean_matrix(df_clbp_centrality)
    # ### Networkx graph of degree centrality of commit2_weights.csv nodes filtered by scilpy
    # networkx_graph_convertor(df_clbp_mean_filter, df_clbp_mean_centrality)

    """
    To do NBS analysis of clbp and con Commit2_weights.csv and store results in NBS_results
    """    
    # pval, adj, null = nbs_data(df_con_v1, df_clbp_v1, save_path='/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/23-07-11_v1_')
    # adj_array = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/23-07-11_v1_adj.npy')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/adj_array.txt', adj_array, fmt='%1.3f')
    # null_array = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/23-07-11_v1_null.npy')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/null_array.txt', null_array, fmt='%1.3f')
    # pval_array = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/23-07-11_v1_pval.npy')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/pval_array.txt', pval_array, fmt='%1.3f')

    """
    To create a network graph of z-score connectivity of Commit2_weights.csv of clbp and con at v1 after scilpy filtering
    """        
    # ### Scilpy filter on commit2_weights data
    # mask_clbp_commit2_v2 = df_clbp_v3.groupby('subject').apply(lambda x:scilpy_filter(x))
    # mask_con_commit2_v2 = df_con_v3.groupby('subject').apply(lambda x:scilpy_filter(x))
    
    # ### Calculate z-score
    # df_z_score_v2 = z_score(mask_con_commit2_v2, mask_clbp_commit2_v2)

    # ### Load figure  
    # figure_data = prepare_data(df_z_score_v2, absolute=False)
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/z_score_v3.txt', figure_data.flatten(), fmt='%1.3f') 
    # plot_network(figure_data, load_brainnetome_centroids())

    """
    To create a network graph of z-score connectivity of Commit2_weights.csv of clbp at v1 and v2 after scilpy filtering
    """   
    # ### Scilpy filter on commit2_weights data
    # mask_clbp_commit2_v1 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x))
    # mask_clbp_commit2_v2 = df_clbp_v2.groupby('subject').apply(lambda x:scilpy_filter(x))
    # mask_clbp_commit2_v1.index.names = ['subject', 'roi']
    # mask_clbp_commit2_v2.index.names = ['subject', 'roi']
    # df_clean_v1 = data_cleaner(mask_clbp_commit2_v1)
    # df_clean_v2 = data_cleaner(mask_clbp_commit2_v2)
    # ### Calculate paired t-tests for each ROI
    # merged_df = pd.merge(df_clean_v1, df_clean_v2, on=['subject', 'roi'], suffixes=('_v1', '_v2'))
    # results = {}
    # for roi, data in merged_df.groupby('roi'):
    #     t_statistic, p_value = ttest_rel(data['centrality_v1'], data['centrality_v2'])
    #     results[roi] = {'t_statistic': t_statistic, 'p_value': p_value}
    # ### Create a new DataFrame to store the t-test results
    # df_t_test_results = pd.DataFrame(results).T
    # print(df_t_test_results)
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/t_test_results.txt', df_t_test_results['t_statistic'], fmt='%1.5f')
    # ### Load numpy array of v1 and v2 and calculate delta
    # delta_v2_v1 = mask_clbp_commit2_v2 - mask_clbp_commit2_v1
    # ### Mean of mask_clbp_commit2 connectivity matrix for df_connectivity_matrix of networkx_graph_convertor
    # df_clbp_mean_filter = mean_matrix(mask_clbp_commit2_v1)
    # df_clbp_mean_centrality = mean_matrix(delta_v2_v1)
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/delta_v2_v1.txt', df_clbp_mean_centrality, fmt='%1.5f')
    
    # ### Load figure
    # plot_network(delta_v2_v1, load_brainnetome_centroids())

if __name__ == "__main__":
    main()