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
import pingouin as pg
from scipy.stats import ttest_rel
from scipy.stats import t
from statsmodels.stats.anova import AnovaRM
from connectivity_figures import plot_network
from connectivity_filtering import distance_dependant_filter
from connectivity_filtering import load_brainnetome_centroids
from connectivity_filtering import scilpy_filter
from connectivity_filtering import threshold_filter
from connectivity_graphing import circle_graph
from connectivity_graphing import histogram
from connectivity_stats import friedman
from connectivity_stats import icc
from connectivity_stats import mean_matrix
from connectivity_stats import nbs_data
from connectivity_stats import paired_t_test
from connectivity_stats import z_score
from graph_theory_functions import networkx_betweenness_centrality
from graph_theory_functions import networkx_cluster_coefficient
from graph_theory_functions import networkx_degree_centrality
from graph_theory_functions import networkx_eigenvector_centrality
from graph_theory_functions import networkx_graph_convertor
from graph_theory_functions import z_score_centrality
from connectivity_read_files import find_files_with_common_name

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
    subjects_to_remove = ['sub-pl008', 'sub-pl016', 'sub-pl037', 'sub-pl039'] # Need to hard-code this!
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
    To create a Networkx graph of delta betweenness centrality of clbp Commit2_weights.csv of clbp at v1, v2 and v3 after scilpy filtering
    Calculates ICC, Friedman, paired t-test for every ROI, calculates delta
    """
    ### Scilpy filter on commit2_weights data
    mask_clbp_commit2_v1 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
    mask_clbp_commit2_v2 = df_clbp_v2.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
    mask_clbp_commit2_v3 = df_clbp_v3.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
    ### Degree centrality
    df_clbp_centrality_v1 = mask_clbp_commit2_v1.groupby('subject').apply(lambda x:networkx_betweenness_centrality(x)).rename(columns={0: 'centrality'})
    df_clbp_centrality_v1.index.names = ['subject', 'roi']
    df_clean_centrality_v1 = data_cleaner(df_clbp_centrality_v1)
    df_clbp_centrality_v2 = mask_clbp_commit2_v2.groupby('subject').apply(lambda x:networkx_betweenness_centrality(x)).rename(columns={0: 'centrality'})
    df_clbp_centrality_v2.index.names = ['subject', 'roi']
    df_clean_centrality_v2 = data_cleaner(df_clbp_centrality_v2)
    df_clbp_centrality_v3 = mask_clbp_commit2_v3.groupby('subject').apply(lambda x:networkx_betweenness_centrality(x)).rename(columns={0: 'centrality'})
    df_clbp_centrality_v3.index.names = ['subject', 'roi']
    df_clean_centrality_v3 = data_cleaner(df_clbp_centrality_v3)
    
    # ### Calculate ICC
    # results = icc(df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3)
    
    ### Calculate Friedman test
    stat, pval = friedman(df_clean_centrality_v1, df_clean_centrality_v2, df_clean_centrality_v3)
    stat.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_stat.csv')
    pval.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_pval.csv')
    # ### Calculate paired t-tests for each ROI
    # t_result_v2_v1 = paired_t_test(df_clean_centrality_v1, df_clean_centrality_v2)
    # t_result_v3_v2 = paired_t_test(df_clean_centrality_v2, df_clean_centrality_v3)

    # ### Load numpy array of v1, v2 and v3 and calculate delta
    # delta_v2_v1 = df_clean_centrality_v2 - df_clean_centrality_v1
    # delta_v3_v2 = df_clean_centrality_v3 - df_clean_centrality_v2
    ### Mean of mask_clbp_commit2 connectivity matrix for df_connectivity_matrix of networkx_graph_convertor
    df_mean_filter = mean_matrix(mask_clbp_commit2_v1)
    df_friedman_stat = mean_matrix(stat)
    df_friedman_stat = df_friedman_stat.reset_index()
    df_friedman_pval = mean_matrix(pval)
    # df_mean_delta_v2v1 = mean_matrix(delta_v2_v1)
    # df_mean_delta_v3v2 = mean_matrix(delta_v3_v2)
    # df_mean_v1 = mean_matrix(df_clean_centrality_v1)
    # df_mean_v2 = mean_matrix(df_clean_centrality_v2)
    # df_mean_v3 = mean_matrix(df_clean_centrality_v3)
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_stat.txt', df_friedman_stat, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_pval.txt', df_friedman_pval, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/delta_v2_v1.txt', df_mean_delta_v2v1, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/delta_v3_v2.txt', df_mean_delta_v3v2, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/mean_v1.txt', df_mean_v1, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/mean_v2.txt', df_mean_v2, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/mean_v3.txt', df_mean_v3, fmt='%1.5f')
    ### Networkx graph of degree centrality of commit2_weights.csv nodes filtered by scilpy
    networkx_graph_convertor(df_mean_filter, df_friedman_stat)
    # networkx_graph_convertor(df_mean_filter, t_result_v2_v1)
    # networkx_graph_convertor(df_mean_filter, t_result_v3_v2)

    """
    To create a Networkx graph of z-score cluster coefficient of Commit2_weights.csv of clbp and con at v1 after scilpy filtering
    """
    # ### Scilpy filter on commit2_weights data
    # mask_clbp_commit2 = df_clbp_v2.groupby('subject').apply(lambda x:scilpy_filter(x, 'v1'))
    # mask_con_commit2 = df_con_v2.groupby('subject').apply(lambda x:scilpy_filter(x, 'v1'))
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
    
    # ### Networkx graph of degree centrality of commit2_weights.csv nodes filtered by scilpy
    # networkx_graph_convertor(df_clbp_mean_filter, df_z_score_nx)

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
    # mask_clbp_commit2_v2 = df_clbp_v2.groupby('subject').apply(lambda x:scilpy_filter(x, 'v2'))
    # mask_con_commit2_v2 = df_con_v2.groupby('subject').apply(lambda x:scilpy_filter(x, 'v2'))
    
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
    # mask_clbp_commit2_v3 = df_clbp_v3.groupby('subject').apply(lambda x:scilpy_filter(x))
    # mask_clbp_commit2_v1.index.names = ['subject', 'unused', 'roi']
    # mask_clbp_commit2_v2.index.names = ['subject', 'unused', 'roi']
    # mask_clbp_commit2_v3.index.names = ['subject', 'unused', 'roi']
    # df_reset_v1 = mask_clbp_commit2_v1.droplevel(level='unused')
    # df_reset_v2 = mask_clbp_commit2_v2.droplevel(level='unused')
    # df_reset_v3 = mask_clbp_commit2_v3.droplevel(level='unused')
    # df_clean_v1 = data_cleaner(df_reset_v1)
    # df_clean_v2 = data_cleaner(df_reset_v2)
    # df_clean_v3 = data_cleaner(df_reset_v3)
    # ### Load numpy array of v1 and v2 and calculate delta
    # delta_v3_v2 = df_clean_v3 - df_clean_v2
    # delta_v2_v1 = df_clean_v2 - df_clean_v1
    # ### Mean of mask_clbp_commit2 connectivity matrix for df_connectivity_matrix of networkx_graph_convertor
    # df_clbp_mean_filter = mean_matrix(df_clean_v1)
    # df_clbp_mean_v2 = mean_matrix(df_clean_v2)
    # df_clbp_mean_v3 = mean_matrix(df_clean_v3)
    # df_clbp_mean_delta = mean_matrix(delta_v2_v1)
    # df_v1 = prepare_data(df_clbp_mean_filter)
    # df_v2 = prepare_data(df_clbp_mean_v2)
    # df_v3 = prepare_data(df_clbp_mean_v3)
    # ### Load figure
    # figure_data = prepare_data(df_clbp_mean_delta, absolute=False)
    
    # plot_network(figure_data, load_brainnetome_centroids())

if __name__ == "__main__":
    main()