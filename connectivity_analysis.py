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
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import t
from statsmodels.stats.anova import AnovaRM
from connectivity_figures import plot_network
from connectivity_filtering import distance_dependant_filter, load_brainnetome_centroids, scilpy_filter, threshold_filter, sex_filter
from connectivity_graphing import circle_graph, histogram
from connectivity_read_files import find_files_with_common_name
from bct_graph_theory_fct import  compute_betweenness, compute_cluster, compute_degree, compute_eigenvector, compute_global_efficiency, compute_small_world
from connectivity_stats import mean_matrix, z_score, friedman, nbs_data
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

def data_cleaner(df_connectivity_matrix, condition=int):
    """
    Returns Data with only the desired patients for centrality measures, as some have missing values. Used to obtain 
    results of different graph theory metrics for comparison accross visits.

    Parameters
    ----------
    df_connectivity_matrix : (NxS rows, 1 column) pandas DataFrame

    Returns
    -------
    df_connectivity_matrix : (NxS-X, 1) pandas DataFrame
    """
    df_connectivity_matrix.index.names = ['subject', 'roi']
    subjects_to_remove_clbp = ['sub-pl008', 'sub-pl016', 'sub-pl037', 'sub-pl039'] 
    subjects_to_remove_con = ['sub-pl004']
    if condition == 'con':
        df_cleaned = df_connectivity_matrix[~df_connectivity_matrix.index.get_level_values('subject').isin(subjects_to_remove_con)]
    if condition == 'clbp':
        df_cleaned = df_connectivity_matrix[~df_connectivity_matrix.index.get_level_values('subject').isin(subjects_to_remove_clbp)]

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
    To calculate global efficiency and small-worldness of connectivity matrices after scilpy filtering
    """
    mask_clbp_v1 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x, 'v1'))
    # global_efficiency_clbp_v1 = mask_clbp_v1.groupby('subject').apply(lambda x:compute_global_efficiency(x))
    # mean_global_efficiency_clbp_v1 = global_efficiency_clbp_v1.mean()
    # print(mean_global_efficiency_clbp_v1)
    # mask_con_v1 = df_con_v1.groupby('subject').apply(lambda x:scilpy_filter(x, 'v1'))
    # global_efficiency_con_v1 = mask_con_v1.groupby('subject').apply(lambda x:compute_global_efficiency(x))
    # mean_global_efficiency_con_v1 = global_efficiency_con_v1.mean()
    # print(mean_global_efficiency_con_v1)
    # t_test = ttest_ind(global_efficiency_clbp_v1, global_efficiency_con_v1)
    
    
    """
    To calculate z-score of graph theory metrics analysis with bct based on sex
    """
    # ### Fetch data
    # centrality_clbp_v1 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v1_scilpy(v1).csv', index_col=['subject', 'roi'])
    # centrality_con_v1 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v1_scilpy(v1).csv', index_col=['subject', 'roi'])
    # centrality_clbp_v2 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v2_scilpy(v2).csv', index_col=['subject', 'roi'])
    # centrality_con_v2 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v2_scilpy(v2).csv', index_col=['subject', 'roi'])
    # centrality_clbp_v3 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v3_scilpy(v3).csv', index_col=['subject', 'roi'])
    # centrality_con_v3 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v3_scilpy(v3).csv', index_col=['subject', 'roi'])
    
    # ### Apply sex filter
    # f_clbp_v1 = sex_filter(centrality_clbp_v1, sex='M', condition='clbp')
    # f_con_v1 = sex_filter(centrality_con_v1, sex='M', condition='con')
    # f_clbp_v2 = sex_filter(centrality_clbp_v2, sex='M', condition='clbp')
    # f_con_v2 = sex_filter(centrality_con_v2, sex='M', condition='con')
    # f_clbp_v3 = sex_filter(centrality_clbp_v3, sex='M', condition='clbp')
    # f_con_v3 = sex_filter(centrality_con_v3, sex='M', condition='con')
    
    # ### Calculate z-score
    # z_score_v1 = z_score(f_con_v1, f_clbp_v1)
    # z_score_v2 = z_score(f_con_v2, f_clbp_v2)
    # z_score_v3 = z_score(f_con_v3, f_clbp_v3)
    
    # ### Save z-score in txt file
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/z_score_v1.txt', z_score_v1, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/z_score_v2.txt', z_score_v2, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/z_score_v3.txt', z_score_v3, fmt='%1.5f')
    
    """
    Calculate Friedman test of graph theory metrics analysis with bct based on sex
    """
    # ### Fetch data
    # centrality_clbp_v1f = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v1_scilpy(all).csv')
    # centrality_con_v1f = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v1_scilpy(all).csv')
    # centrality_clbp_v2f = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v2_scilpy(all).csv')
    # centrality_con_v2f = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v2_scilpy(all).csv')
    # centrality_clbp_v3f = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v3_scilpy(all).csv')
    # centrality_con_v3f = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v3_scilpy(all).csv')

    # ### Apply sex filter
    # f_clbp_v1 = sex_filter(centrality_clbp_v1f, sex='M', condition='clbp')
    # f_con_v1 = sex_filter(centrality_con_v1f, sex='M', condition='con')
    # f_clbp_v2 = sex_filter(centrality_clbp_v2f, sex='M', condition='clbp')
    # f_con_v2 = sex_filter(centrality_con_v2f, sex='M', condition='con')
    # f_clbp_v3 = sex_filter(centrality_clbp_v3f, sex='M', condition='clbp')
    # f_con_v3 = sex_filter(centrality_con_v3f, sex='M', condition='con')

    # ### Calculate Friedman test
    # stat_con, pval_con = friedman(f_con_v1, f_con_v2, f_con_v3)
    # df_con_stat = mean_matrix(stat_con)
    # df_con_pval = mean_matrix(pval_con)
    # stat_clbp, pval_clbp = friedman(f_clbp_v1, f_clbp_v2, f_clbp_v3)
    # df_clbp_stat = mean_matrix(stat_clbp)
    # df_clbp_pval = mean_matrix(pval_clbp)
    
    # ### Save Friedman test in txt file
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_stat_con.txt', df_con_stat, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_pval_con.txt', df_con_pval, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_stat.txt', df_clbp_stat, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_pval.txt', df_clbp_pval, fmt='%1.5f')
    
    """
    To create a Networkx graph of z_score graph theory metrics at v1, v2 and v3 after scilpy filtering
    """
    # ### Fetch data
    # centrality_clbp_v1 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v1_scilpy(v1).csv', index_col=['subject', 'roi'])
    # centrality_con_v1 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v1_scilpy(v1).csv', index_col=['subject', 'roi'])
    # centrality_clbp_v2 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v2_scilpy(v2).csv', index_col=['subject', 'roi'])
    # centrality_con_v2 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v2_scilpy(v2).csv', index_col=['subject', 'roi'])
    # centrality_clbp_v3 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v3_scilpy(v3).csv', index_col=['subject', 'roi'])
    # centrality_con_v3 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v3_scilpy(v3).csv', index_col=['subject', 'roi'])

    # ### Calculate z-score
    # z_score_v1 = z_score(centrality_con_v1, centrality_clbp_v1)
    # z_score_v2 = z_score(centrality_con_v2, centrality_clbp_v2)
    # z_score_v3 = z_score(centrality_con_v3, centrality_clbp_v3)

    # ### Obtain connectivity matrix to graph
    # mask_clbp_commit2_v1 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
    # df_mean_filter = mean_matrix(mask_clbp_commit2_v1)
    # networkx_graph_convertor(df_mean_filter, z_score_v1)
    # networkx_graph_convertor(df_mean_filter, z_score_v2)
    # networkx_graph_convertor(df_mean_filter, z_score_v3)
    """
    To create a Networkx graph of delta betweenness centrality of clbp Commit2_weights.csv of clbp at v1, v2 and v3 after scilpy filtering
    Calculates ICC, Friedman, paired t-test for every ROI, calculates delta
    """
    # ### Calculate ICC
    # results_my_icc = my_icc(df_clean_clbp_v1, df_clean_clbp_v2, df_clean_clbp_v3)
    # results_icc = icc(df_clean_clbp_v1, df_clean_clbp_v2, df_clean_clbp_v3)
    
    ### Calculate Friedman test
    # stat, pval = friedman(df_clean_con_v1, df_clean_con_v2, df_clean_con_v3)
    
    # ### Calculate paired t-tests for each ROI
    # t_result_v2_v1 = paired_t_test(df_clean_clbp_v1, df_clean_clbp_v2)
    # t_result_v3_v2 = paired_t_test(df_clean_clbp_v2, df_clean_clbp_v3)

    # ### Calculate difference between visits
    # delta_v2_v1 = df_clean_clbp_v2 - df_clean_clbp_v1
    # delta_v3_v2 = df_clean_clbp_v3 - df_clean_clbp_v2

    # ### Mean of mask_clbp_commit2 connectivity matrix for df_connectivity_matrix of networkx_graph_convertor
    # df_mean_filter = mean_matrix(mask_clbp_commit2_v1)
    # df_friedman_stat = mean_matrix(stat)
    # df_friedman_pval = mean_matrix(pval)
    # df_mean_delta_v2v1 = mean_matrix(delta_v2_v1)
    # df_mean_delta_v3v2 = mean_matrix(delta_v3_v2)
    # df_mean_v1 = mean_matrix(df_clean_con_v1)
    # df_mean_v2 = mean_matrix(df_clean_con_v2)
    # df_mean_v3 = mean_matrix(df_clean_con_v3)
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_stat.txt', df_friedman_stat, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_pval.txt', df_friedman_pval, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/delta_v2_v1.txt', df_mean_delta_v2v1, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/delta_v3_v2.txt', df_mean_delta_v3v2, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/mean_v1.txt', df_mean_v1, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/mean_v2.txt', df_mean_v2, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/mean_v3.txt', df_mean_v3, fmt='%1.5f')
    # ### Networkx graph of degree centrality of commit2_weights.csv nodes filtered by scilpy
    # networkx_graph_convertor(df_mean_filter, df_friedman_stat)
    # networkx_graph_convertor(df_mean_filter, t_result_v2_v1)
    # networkx_graph_convertor(df_mean_filter, t_result_v3_v2)

    """
    To do NBS analysis of clbp and con Commit2_weights.csv and store results in NBS_results
    """    
    # pval, adj, null = nbs_data(df_con_v1, df_clbp_v1, save_path='/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/male/23-07-11_v1_')
    # adj_array = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/male/23-07-11_v1_adj.npy')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/male/adj_array.txt', adj_array, fmt='%1.3f')
    # null_array = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/male/23-07-11_v1_null.npy')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/male/null_array.txt', null_array, fmt='%1.3f')
    # pval_array = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/male/23-07-11_v1_pval.npy')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_nbs/male/pval_array.txt', pval_array, fmt='%1.3f')

    """
    To create a network graph of z-score connectivity of Commit2_weights.csv of clbp and con at v1 after scilpy and sex filtering
    """        
    # ### Scilpy filter on commit2_weights data
    # mask_clbp_commit2_v1 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x, 'v1'))
    # mask_con_commit2_v1 = df_con_v1.groupby('subject').apply(lambda x:scilpy_filter(x, 'v1'))
    # mask_con_commit2_v1.index.names = ['subject', 'unused', 'roi']
    # mask_clbp_commit2_v1.index.names = ['subject', 'unused', 'roi']
    # df_con_vf = mask_con_commit2_v1.droplevel(level='unused')
    # df_clbp_vf = mask_clbp_commit2_v1.droplevel(level='unused')

    # ### Apply sex filter
    # f_clbp_v1 = sex_filter(df_clbp_vf, sex='F', condition='clbp')
    # f_con_v1 = sex_filter(df_con_vf, sex='F', condition='con')
    # m_clbp_v1 = sex_filter(df_clbp_vf, sex='M', condition='clbp')
    # m_con_v1 = sex_filter(df_con_vf, sex='M', condition='con')
    
    # ### Calculate z-score
    # f_z_score = z_score(f_con_v1, f_clbp_v1)
    # m_z_score = z_score(m_con_v1, m_clbp_v1)

    # ### Load figure  
    # figure_data_f = prepare_data(f_z_score, absolute=False)
    # # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/z_score_f.txt', figure_data_f.flatten(), fmt='%1.3f') 
    # # plot_network(figure_data_f, load_brainnetome_centroids())

    # figure_data_m = prepare_data(m_z_score, absolute=False)
    # # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/z_score_m.txt', figure_data_m.flatten(), fmt='%1.3f') 
    # # plot_network(figure_data_m, load_brainnetome_centroids())

    # circle_graph(figure_data_f)
    """
    To create a network graph of z-score connectivity of Commit2_weights.csv of clbp at v1 and v2 after scilpy filtering
    """   
    # ### Scilpy filter on commit2_weights data
    # mask_clbp_commit2_v1 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
    # mask_clbp_commit2_v2 = df_clbp_v2.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
    # mask_clbp_commit2_v3 = df_clbp_v3.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
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