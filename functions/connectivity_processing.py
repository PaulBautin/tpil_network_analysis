from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse de la connectivite avec commmit2
#
# example: python connectivity_analysis.py -clbp <dir> -con <dir>
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
from functions.connectivity_read_files import find_files_with_common_name
from functions.connectivity_filtering import scilpy_filter, distance_dependant_filter, threshold_filter, sex_filter




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
    # df_connectivity_matrix.index = pd.MultiIndex.from_frame(df_connectivity_matrix[['subject', 'roi']])
    subjects_to_remove_clbp = ['sub-pl008', 'sub-pl016', 'sub-pl037', 'sub-pl039'] 
    subjects_to_remove_con = ['sub-pl004']
    if condition == 'con':
        df_cleaned = df_connectivity_matrix[~df_connectivity_matrix.index.get_level_values(0).isin(subjects_to_remove_con)]
    if condition == 'clbp':
        df_cleaned = df_connectivity_matrix[~df_connectivity_matrix.index.get_level_values(0).isin(subjects_to_remove_clbp)]

    return df_cleaned


def data_processor(df_connectivity_matrix, session='v1', condition='clbp', sex=None, filter=None, clean=False):
    # Validate input
    valid_sessions = ['v1', 'v2', 'v3', 'all']
    if session not in valid_sessions:
        print("Invalid session. Valid sessions:", valid_sessions)
        return None
    valid_conditions = ['clbp','con']
    if condition not in valid_conditions:
        print("Invalid condition. Valid conditions:", valid_conditions)
    valid_sex = ['M', 'F', None]
    if sex not in valid_sex:
        print("Invalid sex. Valid sex:", valid_sex)
        return None
    valid_filters = ['scilpy', 'threshold', 'distance_dependant', None]
    if filter not in valid_filters:
        print("Invalid filter. Valid filters:", valid_filters)
        return None
    
    # Apply requested filters
    if filter == 'scilpy' and session =='v1':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:scilpy_filter(x, 'v1'))
    elif filter == 'scilpy' and session =='v2':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:scilpy_filter(x, 'v2'))
    elif filter == 'scilpy' and session =='v3':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:scilpy_filter(x, 'v3'))
    elif filter == 'scilpy' and session =='all':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:scilpy_filter(x, 'all'))
    elif filter == 'threshold':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:threshold_filter(x))
    elif filter == 'distance_dependant':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:distance_dependant_filter(x))
    else:
        df_filtered = df_connectivity_matrix
    
    # Apply requested sex
    if sex == "F" and condition =="clbp":
        df_sex_filtered = df_filtered.groupby('subject').apply(lambda x:sex_filter(x, sex='F', condition='clbp'))
    elif sex == "F" and condition =="con":
        df_sex_filtered = df_filtered.groupby('subject').apply(lambda x:sex_filter(x, sex='F', condition='con'))
    elif sex == "M" and condition =="clbp":
        df_sex_filtered = df_filtered.groupby('subject').apply(lambda x:sex_filter(x, sex='M', condition='clbp'))
    elif sex == "M" and condition =="con":
        df_sex_filtered = df_filtered.groupby('subject').apply(lambda x:sex_filter(x, sex='M', condition='con'))
    else:
        df_sex_filtered = df_filtered
    
    # Apply cleaning method
    if clean == True and condition == "clbp":
        df_clean_sex_filtered = data_cleaner(df_sex_filtered, condition='clbp')
    elif clean == True and condition == "con":
        df_clean_sex_filtered = data_cleaner(df_sex_filtered, condition='con')
    else:
        df_clean_sex_filtered = df_sex_filtered
    
    return df_clean_sex_filtered

        
