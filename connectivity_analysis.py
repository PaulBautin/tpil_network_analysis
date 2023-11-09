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
from functions.connectivity_figures import plot_network, circle_graph, histogram
from functions.connectivity_filtering import distance_dependant_filter, load_brainnetome_centroids, scilpy_filter, threshold_filter, sex_filter
from functions.connectivity_processing import data_processor
from functions.connectivity_read_files import find_files_with_common_name
from functions.gtm_bct import  compute_betweenness, compute_cluster, compute_degree, compute_eigenvector, compute_global_efficiency, compute_small_world, compute_shortest_path
from functions.connectivity_stats import mean_matrix, z_score, friedman, nbs_data, my_icc, icc
from functions.gtm_nx import networkx_graph_convertor



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
    
    """ 
    Get connectivity data
    """
    
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

    # ### Select data and filters to apply
    # df_clean_clbp_v1 = data_processor(df_clbp_v1, session='v1', condition='clbp', filter='scilpy')
    
    """
    To calculate z-score and friedman test of graph theory metrics analysis with bct based on sex
    """
    # ### Fetch data
    # centrality_clbp_v1 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v1_scilpy(v1).csv', index_col=['subject', 'roi'])
    # centrality_con_v1 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v1_scilpy(v1).csv', index_col=['subject', 'roi'])
    # centrality_clbp_v2 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v2_scilpy(v2).csv', index_col=['subject', 'roi'])
    # centrality_con_v2 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v2_scilpy(v2).csv', index_col=['subject', 'roi'])
    # centrality_clbp_v3 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/clbp_v3_scilpy(v3).csv', index_col=['subject', 'roi'])
    # centrality_con_v3 = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/betweenness_centrality/bct/con_v3_scilpy(v3).csv', index_col=['subject', 'roi'])
    
    # ### Apply sex filter
    # f_clbp_v1 = sex_filter(centrality_clbp_v1, sex='F', condition='clbp')
    # f_con_v1 = sex_filter(centrality_con_v1, sex='F', condition='con')
    # f_clbp_v2 = sex_filter(centrality_clbp_v2, sex='F', condition='clbp')
    # f_con_v2 = sex_filter(centrality_con_v2, sex='F', condition='con')
    # f_clbp_v3 = sex_filter(centrality_clbp_v3, sex='F', condition='clbp')
    # f_con_v3 = sex_filter(centrality_con_v3, sex='F', condition='con')
    
    # ### Calculate z-score
    # z_score_v1 = z_score(f_con_v1, f_clbp_v1)
    # z_score_v2 = z_score(f_con_v2, f_clbp_v2)
    # z_score_v3 = z_score(f_con_v3, f_clbp_v3)
    
    # ### Save z-score in txt file
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/z_score_v1.txt', z_score_v1, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/z_score_v2.txt', z_score_v2, fmt='%1.5f')
    # np.savetxt('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/z_score_v3.txt', z_score_v3, fmt='%1.5f')
    
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
    
    # ### Obtain connectivity matrix to graph
    # mask_clbp_commit2_v1 = df_clbp_v1.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
    # df_mean_filter = mean_matrix(mask_clbp_commit2_v1)
    # networkx_graph_convertor(df_mean_filter, df_con_stat)
    # networkx_graph_convertor(df_mean_filter, df_clbp_stat)

    """
    To calculate the ICC
    """
    # ### Calculate ICC
    # results_my_icc = my_icc(df_clean_clbp_v1, df_clean_clbp_v2, df_clean_clbp_v3)
    # results_icc = icc(df_clean_clbp_v1, df_clean_clbp_v2, df_clean_clbp_v3)
    

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
    To create a network graph of z-score connectivity of Commit2_weights.csv of clbp and con 
    """        
    
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

if __name__ == "__main__":
    main()