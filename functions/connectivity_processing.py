from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Fonction pour préparer l'analyse de la connectivité structurelle
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
from functions.connectivity_filtering import scilpy_filter, distance_dependant_filter, threshold_filter, sex_filter


def prepare_data(df_connectivity_matrix, absolute=False):
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
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:scilpy_filter(x, 'v1', print_density=False))
    elif filter == 'scilpy' and session =='v2':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:scilpy_filter(x, 'v2', print_density=False))
    elif filter == 'scilpy' and session =='v3':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:scilpy_filter(x, 'v3', print_density=False))
    elif filter == 'scilpy' and session =='all':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:scilpy_filter(x, 'all', print_density=False))
    elif filter == 'threshold':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:threshold_filter(x, print_density=False))
    elif filter == 'distance_dependant':
        df_filtered = df_connectivity_matrix.groupby('subject').apply(lambda x:distance_dependant_filter(x))
    else:
        df_filtered = df_connectivity_matrix.set_index(["subject", "subject","roi"])

    # Reset the index to remove the duplicated 'subject' level
    df_filtered.index = df_filtered.index.droplevel(0)    
    
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

def multiply_matrix(matrix_1, matrix_2):
    df_mult = matrix_1 * matrix_2
    return df_mult

def difference_visits(df):
    """
    A script used to calculate differences between visits
    """
    # Calculate the differences between visits
    df['efficiency_diff'] = df.groupby('subject')['efficiency'].diff(periods=1)

    df['strength_diff'] = df.groupby('subject')['strength'].diff(periods=1)

    df['cluster_diff'] = df.groupby('subject')['cluster'].diff(periods=1)

    df['small_world_diff'] = df.groupby('subject')['small_world'].diff(periods=1)

    df['modularity_diff'] = df.groupby('subject')['modularity'].diff(periods=1)

    return df
