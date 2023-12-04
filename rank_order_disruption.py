from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse du rank order disruption
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
from functions.connectivity_processing import data_processor
from functions.connectivity_read_files import find_files_with_common_name
from functions.connectivity_figures import disruption_index, disruption_index_combined
from functions.gtm_bct import bct_master

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
    ### Get parser elements
    parser = get_parser()
    arguments = parser.parse_args()
    path_results_con = os.path.abspath(os.path.expanduser(arguments.con))
    path_results_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    path_output = os.path.abspath(arguments.o)
    
    """ 
    Get connectivity data
    """
    df_con = find_files_with_common_name(path_results_con, arguments.connectivity_type)
    df_clbp = find_files_with_common_name(path_results_clbp, arguments.connectivity_type)
    ### work only on one session at a time
    df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)
    df_con_v2 = df_con[df_con['session'] == "v2"].drop("session", axis=1)
    df_clbp_v2 = df_clbp[df_clbp['session'] == "v2"].drop("session", axis=1)
    df_con_v3 = df_con[df_con['session'] == "v3"].drop("session", axis=1)
    df_clbp_v3 = df_clbp[df_clbp['session'] == "v3"].drop("session", axis=1)

    """
    Prepare data for analysis
    """
    ### Select data and filters to apply
    df_clean_clbp_v1 = data_processor(df_clbp_v1, session='v1', condition='clbp', filter='scilpy')
    df_clean_con_v1 = data_processor(df_con_v1, session='v1', condition='con', filter='scilpy')
    df_clean_clbp_v2 = data_processor(df_clbp_v2, session='v2', condition='clbp', filter='scilpy')
    df_clean_con_v2 = data_processor(df_con_v2, session='v2', condition='con', filter='scilpy')
    df_clean_clbp_v3 = data_processor(df_clbp_v3, session='v3', condition='clbp', filter='scilpy')
    df_clean_con_v3 = data_processor(df_con_v3, session='v3', condition='con', filter='scilpy')
    

    """
    Compute degree centrality
    """
    centrality_clbp_v1 = df_clean_clbp_v1.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality'))
    centrality_clbp_v2 = df_clean_clbp_v2.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality'))
    centrality_clbp_v3 = df_clean_clbp_v3.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality'))
    centrality_con_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality'))
    centrality_con_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality'))
    centrality_con_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality'))
    
    disruption_index_combined(centrality_con_v1, centrality_clbp_v1)
    # kd1 = disruption_index(centrality_con_v1, centrality_clbp_v1)
    # kd2 = disruption_index(centrality_con_v2, centrality_clbp_v2)
    # kd3 = disruption_index(centrality_con_v3, centrality_clbp_v3)

    # kd1.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_kd/strength_commit_v1.csv')
    # kd2.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_kd/strength_commit_v2.csv')
    # kd3.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/results_kd/strength_commit_v3.csv')

if __name__ == "__main__":
    main()