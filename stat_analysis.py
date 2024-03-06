from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse des statistiques
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
import glob
import bct
import networkx as nx
import pingouin as pg
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import t
from statsmodels.stats.anova import AnovaRM
from functions.connectivity_filtering import distance_dependant_filter, load_brainnetome_centroids, scilpy_filter, threshold_filter, sex_filter, pain_duration_filter, limbic_system_filter
from functions.connectivity_processing import data_processor, prepare_data
from functions.connectivity_read_files import find_files_with_common_name
from functions.connectivity_stats import mean_matrix, z_score, friedman, nbs_data, my_icc, icc, calculate_icc_all_rois, calculate_cv
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
    mandatory.add_argument(
        "-connectivity_type",
        required=True,
        default='connectivity_results',
        help='type of connectivity data to work with (e.g. "commit2_weights.csv")',
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
    # ### Get parser elements
    # parser = get_parser()
    # arguments = parser.parse_args()
    # path_results_con = os.path.abspath(os.path.expanduser(arguments.con))
    # path_results_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    # path_output = os.path.abspath(arguments.o)
    
    """ 
    Get connectivity data
    """
    # df_con = find_files_with_common_name(path_results_con, arguments.connectivity_type)
    # df_clbp = find_files_with_common_name(path_results_clbp, arguments.connectivity_type)
    # ### work only on one session at a time
    # df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    # df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)
    # df_con_v2 = df_con[df_con['session'] == "v2"].drop("session", axis=1)
    # df_clbp_v2 = df_clbp[df_clbp['session'] == "v2"].drop("session", axis=1)
    # df_con_v3 = df_con[df_con['session'] == "v3"].drop("session", axis=1)
    # df_clbp_v3 = df_clbp[df_clbp['session'] == "v3"].drop("session", axis=1)
    
    ### Fetch graph theory metrics data
    gtm_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_metrics_con.csv', index_col=['subject', 'roi'])
    gtm_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_metrics.csv', index_col=['subject', 'roi'])
    
    gtm_limb_con = limbic_system_filter(gtm_con)
    gtm_limb_clbp = limbic_system_filter(gtm_clbp)

    gtm_global_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_global_metrics_con.csv', index_col=['subject'])
    gtm_global_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_global_metrics.csv', index_col=['subject'])
    

    """
    Perform Friedman test of chosen metric 
    """
    ### Calculate Friedman test for degree
    stat_con_d, pval_con_d = friedman(gtm_con, roi_column='roi', metric_column='degree')
    degree_con = pd.concat([stat_con_d, pval_con_d], axis=1)
    degree_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/degree_con.csv')
    stat_clbp_d, pval_clbp_d = friedman(gtm_clbp, roi_column='roi', metric_column='degree')
    degree_clbp = pd.concat([stat_clbp_d, pval_clbp_d], axis=1)
    degree_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/degree_clbp.csv')
    ### Calculate Friedman test for strength
    stat_con_s, pval_con_s = friedman(gtm_con, roi_column='roi', metric_column='strength')
    strength_con = pd.concat([stat_con_s, pval_con_s], axis=1)
    strength_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/strength_con.csv')
    stat_clbp_s, pval_clbp_s = friedman(gtm_clbp, roi_column='roi', metric_column='strength')
    strength_clbp = pd.concat([stat_clbp_s, pval_clbp_s], axis=1)
    strength_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/strength_clbp.csv')
    ### Calculate Friedman test for betweenness
    stat_con_b, pval_con_b = friedman(gtm_con, roi_column='roi', metric_column='betweenness')
    betweenness_con = pd.concat([stat_con_b, pval_con_b], axis=1)
    betweenness_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/betweenness_con.csv')
    stat_clbp_b, pval_clbp_b = friedman(gtm_clbp, roi_column='roi', metric_column='betweenness')
    betweenness_clbp = pd.concat([stat_clbp_b, pval_clbp_b], axis=1)
    betweenness_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/betweenness_clbp.csv')
    ### Calculate Friedman test for cluster
    stat_con_c, pval_con_c = friedman(gtm_con, roi_column='roi', metric_column='cluster')
    cluster_con = pd.concat([stat_con_c, pval_con_c], axis=1)
    cluster_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/cluster_con.csv')
    stat_clbp_c, pval_clbp_c = friedman(gtm_clbp, roi_column='roi', metric_column='cluster')
    cluster_clbp = pd.concat([stat_clbp_c, pval_clbp_c], axis=1)
    cluster_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/cluster_clbp.csv')
    ### Calculate Friedman test for efficiency
    stat_con_e, pval_con_e = friedman(gtm_con, roi_column='roi', metric_column='efficiency')
    efficiency_con = pd.concat([stat_con_e, pval_con_e], axis=1)
    efficiency_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/efficiency_con.csv')
    stat_clbp_e, pval_clbp_e = friedman(gtm_clbp, roi_column='roi', metric_column='efficiency')
    efficiency_clbp = pd.concat([stat_clbp_e, pval_clbp_e], axis=1)
    efficiency_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/local/efficiency_clbp.csv')
    
    """
    Perform Friedman test of chosen metric for limbic system
    """
    ### Calculate Friedman test for degree
    stat_con_d, pval_con_d = friedman(gtm_limb_con, roi_column='label', metric_column='degree')
    degree_con = pd.concat([stat_con_d, pval_con_d], axis=1)
    degree_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/degree_con.csv')
    stat_clbp_d, pval_clbp_d = friedman(gtm_limb_clbp, roi_column='label', metric_column='degree')
    degree_clbp = pd.concat([stat_clbp_d, pval_clbp_d], axis=1)
    degree_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/degree_clbp.csv')
    ### Calculate Friedman test for strength
    stat_con_s, pval_con_s = friedman(gtm_limb_con, roi_column='label', metric_column='strength')
    strength_con = pd.concat([stat_con_s, pval_con_s], axis=1)
    strength_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/strength_con.csv')
    stat_clbp_s, pval_clbp_s = friedman(gtm_limb_clbp, roi_column='label', metric_column='strength')
    strength_clbp = pd.concat([stat_clbp_s, pval_clbp_s], axis=1)
    strength_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/strength_clbp.csv')
    ### Calculate Friedman test for betweenness
    stat_con_b, pval_con_b = friedman(gtm_limb_con, roi_column='label', metric_column='betweenness')
    betweenness_con = pd.concat([stat_con_b, pval_con_b], axis=1)
    betweenness_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/betweenness_con.csv')
    stat_clbp_b, pval_clbp_b = friedman(gtm_limb_clbp, roi_column='label', metric_column='betweenness')
    betweenness_clbp = pd.concat([stat_clbp_b, pval_clbp_b], axis=1)
    betweenness_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/betweenness_clbp.csv')
    ### Calculate Friedman test for cluster
    stat_con_c, pval_con_c = friedman(gtm_limb_con, roi_column='label', metric_column='cluster')
    cluster_con = pd.concat([stat_con_c, pval_con_c], axis=1)
    cluster_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/cluster_con.csv')
    stat_clbp_c, pval_clbp_c = friedman(gtm_limb_clbp, roi_column='label', metric_column='cluster')
    cluster_clbp = pd.concat([stat_clbp_c, pval_clbp_c], axis=1)
    cluster_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/cluster_clbp.csv')
    ### Calculate Friedman test for efficiency
    stat_con_e, pval_con_e = friedman(gtm_limb_con, roi_column='label', metric_column='efficiency')
    efficiency_con = pd.concat([stat_con_e, pval_con_e], axis=1)
    efficiency_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/efficiency_con.csv')
    stat_clbp_e, pval_clbp_e = friedman(gtm_limb_clbp, roi_column='label', metric_column='efficiency')
    efficiency_clbp = pd.concat([stat_clbp_e, pval_clbp_e], axis=1)
    efficiency_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/efficiency_clbp.csv')

    """
    Perform Friedman test of chosen metric for global metrics
    """
    ### Calculate Friedman test for efficiency
    stat_con_e, pval_con_e = friedman(gtm_global_con, roi_column=None, metric_column='efficiency')
    efficiency_con = pd.concat([stat_con_e, pval_con_e], axis=1)
    efficiency_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/efficiency_con.csv')
    stat_clbp_e, pval_clbp_e = friedman(gtm_global_clbp, roi_column=None, metric_column='efficiency')
    efficiency_clbp = pd.concat([stat_clbp_e, pval_clbp_e], axis=1)
    efficiency_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/efficiency_clbp.csv')
    ## Calculate Friedman test for strength
    stat_con_s, pval_con_s = friedman(gtm_global_con, roi_column=None, metric_column='strength')
    strength_con = pd.concat([stat_con_s, pval_con_s], axis=1)
    strength_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/strength_con.csv')
    stat_clbp_s, pval_clbp_s = friedman(gtm_global_clbp, roi_column=None, metric_column='strength')
    strength_clbp = pd.concat([stat_clbp_s, pval_clbp_s], axis=1)
    strength_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/strength_clbp.csv')
    ### Calculate Friedman test for cluster
    stat_con_c, pval_con_c = friedman(gtm_global_con, roi_column=None, metric_column='cluster')
    cluster_con = pd.concat([stat_con_c, pval_con_c], axis=1)
    cluster_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/cluster_con.csv')
    stat_clbp_c, pval_clbp_c = friedman(gtm_global_clbp, roi_column=None, metric_column='cluster')
    cluster_clbp = pd.concat([stat_clbp_c, pval_clbp_c], axis=1)
    cluster_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/cluster_clbp.csv')
    ### Calculate Friedman test for small_world
    stat_con_sw, pval_con_sw = friedman(gtm_global_con, roi_column=None, metric_column='small_world')
    small_world_con = pd.concat([stat_con_sw, pval_con_sw], axis=1)
    small_world_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/small_world_con.csv')
    stat_clbp_sw, pval_clbp_sw = friedman(gtm_global_clbp, roi_column=None, metric_column='small_world')
    small_world_clbp = pd.concat([stat_clbp_sw, pval_clbp_sw], axis=1)
    small_world_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/small_world_clbp.csv')
    ### Calculate Friedman test for modularity
    stat_con_m, pval_con_m = friedman(gtm_global_con, roi_column=None, metric_column='modularity')
    modularity_con = pd.concat([stat_con_m, pval_con_m], axis=1)
    modularity_con.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/modularity_con.csv')
    stat_clbp_m, pval_clbp_m = friedman(gtm_global_clbp, roi_column=None, metric_column='modularity')
    modularity_clbp = pd.concat([stat_clbp_m, pval_clbp_m], axis=1)
    modularity_clbp.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/modularity_clbp.csv')

    """
    Calculate the ICC of chosen metric
    """
    
    ### Calculate ICC
    # results_my_icc = my_icc(small_world_v1, small_world_v2, small_world_v3)
    # results_icc = calculate_icc_all_rois(df_age, metric='Score_POQ_total')