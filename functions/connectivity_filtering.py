from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Fonctions pour filtrer la connectivité structurelle
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
from netneurotools.utils import get_centroids
from netneurotools.networks import threshold_network, struct_consensus
from scipy.spatial.distance import squareform, pdist

def load_brainnetome_centroids(image="/Users/Marc-Antoine/Documents/tpil_network_analysis/labels/sub-pl007_ses-v1__nativepro_seg_all.nii.gz"):
    """
    Loads Euclidean coordinates of nodes
    Parameters
    ----------
    image : nifti label image

    Returns
    -------
    bn_centroids : (3, N) np.array node centroid coordinates
    """
    bn_centroids = get_centroids(image)
    return bn_centroids

def distance_dependant_filter(df_connectivity_matrix):
    """
    Calculates distance-dependent group consensus structural connectivity graph
    This filter is considered distance-dependent as it will divide data into bins (based on edge length percentile).
    It then takes the edge with the best consensus (edge represented with the most stability accross subjects)

    Betzel, R. F., Griffa, A., Hagmann, P., & Mišić, B. (2018). Distance- dependent consensus thresholds for generating 
    group-representative structural brain networks. Network Neuroscience, 1-22.

    Parameters
    ----------
    df_connectivity_matrix : (N, NxS) connectivity matrix pandas Dataframe, where N is the number of nodes and S the number of subjects
    Returns
    -------
    consensus_conn : (N, N) binary np.array matrix mask
    """
    # transform to 3d numpy array (N, N, S) with N nodes and S subjects
    np_con_v1 = np.dstack(list(df_connectivity_matrix.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    bn_centroids = load_brainnetome_centroids()
    eu_distance = squareform(pdist(bn_centroids, metric="euclidean"))
    # pairwise distance matrix, (N, 1) vector identifying right hemisphere (0)and left hemisphere (1) first has 8 on left and 7 on right
    hemiid = np.append(np.append(np.arange(0, 210) % 2, np.zeros(8)), np.ones(7))
    consensus_conn = struct_consensus(np_con_v1, distance=eu_distance, hemiid=hemiid.reshape(-1, 1))
    print("\naverage_distance_without_filter: {}".format(np.mean(eu_distance, where=(eu_distance != 0))))
    print("average_distance_with_filter: {}".format(np.mean(eu_distance * consensus_conn, where=(consensus_conn != 0))))
    return consensus_conn


def threshold_matrix(df_connectivity_matrix, **kwargs):
    """
    Uses a minimum spanning tree to ensure that no nodes are disconnected from the resulting thresholded graph.
    Keeps top 10% of connections

    Parameters
    ----------
    df_connectivity_matrix : (N, N) connectivity matrix, pandas Dataframe, percent connection to retain

    Returns
    -------
    threshold_conn : (N, N) binary np.array matrix mask
    """
    # transform to 3d numpy array (N, N, S) with N nodes and S subjects
    np_con_v1 = np.dstack(list(df_connectivity_matrix.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    bn_centroids = load_brainnetome_centroids()
    eu_distance = squareform(pdist(bn_centroids, metric="euclidean"))
    threshold_conn = threshold_network(np.mean(np_con_v1,axis=2), retain=10)
    print_density = kwargs.get('print_density', False)
    if print_density:
        print("\naverage_distance_without_filter: {}".format(np.mean(eu_distance, where=(eu_distance != 0))))
        print("average_distance_with_filter: {}".format(np.mean(eu_distance * threshold_conn, where=(threshold_conn != 0))))

    return threshold_conn

def threshold_filter(df_connectivity_matrix, print_density=True):

    np_threshold = threshold_matrix(df_connectivity_matrix, print_density=print_density)
    df_mult = df_connectivity_matrix.set_index(["subject","roi"])

    mask_data = df_mult * np_threshold
    
    # Calculate link density of matrix before filtering
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    total_possible_connections = (np_connectivity_matrix.shape[0] * (np_connectivity_matrix.shape[0] - 1)) / 2 
    actual_connections = np.count_nonzero(np.triu(np_connectivity_matrix))
    link_density = actual_connections / total_possible_connections * 100
    

    # Calculate link density of matrix after filtering
    np_mask_data = mask_data.to_numpy()
    actual_connections_filtered = np.count_nonzero(np.triu(np_mask_data))
    link_density_filtered = actual_connections_filtered / total_possible_connections * 100
    if print_density:
        print(f"Link Density before filter: {link_density}%")
        print(f"Link Density after filter: {link_density_filtered}%")
    
    return mask_data

#def filter_no_connections(df_connectivity_matrix):
    #df_con_sc = find_files_with_common_name(path_results_con, "sc.csv")
    #df_clbp_sc = find_files_with_common_name(path_results_con, "sc.csv")
    #df_con_sc_v1 = df_con[df_con_sc['session'] == "v1"].drop(["subject", "session"], axis=1)
    #df_clbp_sc_v1 = df_clbp[df_clbp_sc['session'] == "v1"].drop(["subject", "session"], axis=1)
    #df_connectivity_matrix.iloc[:, 1:] = df_connectivity_matrix.iloc[:, 1:].astype(float) # converts in format that np can use
    #df_zero_connections = []
    #df_zero_matrix = np.zeros_like(df_connectivity_matrix.iloc[:, 1:]) 
    #for row in range(df_zero_matrix.shape[0]):
    #    for col in range(df_zero_matrix.shape[1]):
    #        if df_connectivity_matrix.iloc[row, col+1] < 1: # all non-zero values (integers) convert to 1
    #            df_zero_matrix[row, col] = 0
    #            df_zero_connections.append((row, col))
    #        else:
    #            df_zero_matrix[row, col] = 1
    #If mean of ROI < 1, has at least one 0
    #df_mean_matrix = np.mean(df_zero_matrix.reshape(-1, 246, 246), axis=0)
    #df_mean_matrix[df_mean_matrix < 1] = 0
    #return df_mean_matrix

def scilpy_filter(df_connectivity_matrix, session, print_density=False):
    """
    Each edge with a value of 1 represents an edge with at least 90% of the population having at least 1 streamline
    and at least 90% of the population having at least 20mm of average streamlines length. 
    Population is all clbp and control subjects

    Parameters
    ----------
    df_connectivity_matrix :  (N, N) connectivity matrix pandas Dataframe, where N is the number of nodes 

    Returns
    -------
    mask_data : (N, N) filtered conenctivity matrix, pandas DataFrame
    """
    # mask_v1_con_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v1_mask_sc.npy')
    # mask_v1_clbp_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v1_mask_sc.npy')
    # mask_v1_con_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v1_mask_len.npy')
    # mask_v1_clbp_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v1_mask_len.npy')
    # mask_v2_con_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v2_mask_sc.npy')
    # mask_v2_clbp_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v2_mask_sc.npy')
    # mask_v2_con_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v2_mask_len.npy')
    # mask_v2_clbp_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v2_mask_len.npy')
    # mask_v3_con_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v3_mask_sc.npy')
    # mask_v3_clbp_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v3_mask_sc.npy')
    # mask_v3_con_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v3_mask_len.npy')
    # mask_v3_clbp_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v3_mask_len.npy')

    mask_v1_con_sc = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/con_v1_mask_sc.npy')
    mask_v1_clbp_sc = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/clbp_v1_mask_sc.npy')
    mask_v1_con_len = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/con_v1_mask_len.npy')
    mask_v1_clbp_len = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/clbp_v1_mask_len.npy')
    mask_v2_con_sc = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/con_v2_mask_sc.npy')
    mask_v2_clbp_sc = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/clbp_v2_mask_sc.npy')
    mask_v2_con_len = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/con_v2_mask_len.npy')
    mask_v2_clbp_len = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/clbp_v2_mask_len.npy')
    mask_v3_con_sc = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/con_v3_mask_sc.npy')
    mask_v3_clbp_sc = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/clbp_v3_mask_sc.npy')
    mask_v3_con_len = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/con_v3_mask_len.npy')
    mask_v3_clbp_len = np.load('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/scilpy_filters/clbp_v3_mask_len.npy')
    
    df_mult = df_connectivity_matrix.set_index(['subject', 'roi']) # step necessary to apply z-score equation on every subject

    valid_sessions = ['v1', 'v2', 'v3', 'all']
    if session not in valid_sessions:
        print("Invalid session. Valid sessions:", valid_sessions)
        return None
    elif session == 'v1':
        mask_v1 = mask_v1_con_len * mask_v1_clbp_len * mask_v1_con_sc * mask_v1_clbp_sc
        mask_data = df_mult * mask_v1
    elif session == 'v2':
        mask_v2 = mask_v2_con_len * mask_v2_clbp_len * mask_v2_con_sc * mask_v2_clbp_sc
        mask_data = df_mult * mask_v2
    elif session == 'v3':
        mask_v3 = mask_v3_con_len * mask_v3_clbp_len * mask_v3_con_sc * mask_v3_clbp_sc
        mask_data = df_mult * mask_v3
    elif session == 'all':
        mask_all = mask_v1_con_len * mask_v1_clbp_len * mask_v1_con_sc * mask_v1_clbp_sc * mask_v2_con_len * mask_v2_clbp_len * mask_v2_con_sc * mask_v2_clbp_sc * mask_v3_con_len * mask_v3_clbp_len * mask_v3_con_sc * mask_v3_clbp_sc
        mask_data = df_mult * mask_all

    # Calculate link density of matrix before filtering
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    total_possible_connections = (np_connectivity_matrix.shape[0] * (np_connectivity_matrix.shape[0] - 1)) / 2 
    actual_connections = np.count_nonzero(np.triu(np_connectivity_matrix))
    link_density = actual_connections / total_possible_connections * 100
    
    # Calculate link density of matrix after filtering
    np_mask_data = mask_data.to_numpy()
    actual_connections_filtered = np.count_nonzero(np.triu(np_mask_data))
    link_density_filtered = actual_connections_filtered / total_possible_connections * 100
    
    if print_density:
        print(f"Link Density before filter: {link_density}%")
        print(f"Link Density after filter: {link_density_filtered}%")

    return mask_data


def sex_filter(df_connectivity_matrix, sex='F ', condition='clbp'):
    """
    Returns Data with only the desired patients for centrality measures. Used to measure the effect of sex on
    results of different graph theory metrics for comparison across visits.

    Parameters
    ----------
    df_connectivity_matrix : pandas DataFrame
        The DataFrame containing the connectivity matrix.
    sex : str, optional (default is 'M')
        Sex to filter by: 'M' for male, 'F' for female.
    condition : str, optional (default is 'clbp')
        Condition to filter by: 'con' for control, 'clbp' for chronic low back pain.
    index_format : str, optional (default is 'subject_roi')
        The format of the DataFrame's index. It can be 'subject_roi' or 'subject'. 

    Returns
    -------
    df_connectivity_matrix : pandas DataFrame
        The filtered DataFrame.
    """
    # Check if sex is valid
    if sex not in ('M', 'F'):
        raise ValueError("Sex must be 'M' or 'F'")

    # Check if condition is valid
    if condition not in ('con', 'clbp'):
        raise ValueError("Condition must be 'con' or 'clbp'")

    female_clbp_subjects = ['sub-pl007', 'sub-pl016', 'sub-pl017', 'sub-pl019', 'sub-pl027', 'sub-pl031', 'sub-pl035', 'sub-pl038', 'sub-pl039', 'sub-pl042', 'sub-pl049', 'sub-pl056', 'sub-pl064']
    female_con_subjects = ['sub-pl002', 'sub-pl004', 'sub-pl021', 'sub-pl024', 'sub-pl029', 'sub-pl040', 'sub-pl046', 'sub-pl051', 'sub-pl053', 'sub-pl057', 'sub-pl058', 'sub-pl059', 'sub-pl060', 'sub-pl061', 'sub-pl065']
    male_clbp_subjects = ['sub-pl008', 'sub-pl010', 'sub-pl012', 'sub-pl013', 'sub-pl014', 'sub-pl030', 'sub-pl032', 'sub-pl034', 'sub-pl036', 'sub-pl037', 'sub-pl047', 'sub-pl050', 'sub-pl055', 'sub-pl063']
    male_con_subjects = ['sub-pl006', 'sub-pl015', 'sub-pl022', 'sub-pl023', 'sub-pl025', 'sub-pl041', 'sub-pl048', 'sub-pl052', 'sub-pl054', 'sub-pl062']

    # Set 'subject' as the index if it's not already
    if 'subject' not in df_connectivity_matrix.index.names:
        df_connectivity_matrix = df_connectivity_matrix.set_index('subject')
    
    if sex == 'M' and condition == 'clbp':
        df_cleaned = df_connectivity_matrix[df_connectivity_matrix.index.get_level_values('subject').isin(male_clbp_subjects)]
    elif sex == 'M' and condition == 'con':
        df_cleaned = df_connectivity_matrix[df_connectivity_matrix.index.get_level_values('subject').isin(male_con_subjects)]
    elif sex == 'F' and condition == 'clbp':
        df_cleaned = df_connectivity_matrix[df_connectivity_matrix.index.get_level_values('subject').isin(female_clbp_subjects)]
    elif sex == 'F' and condition == 'con':
        df_cleaned = df_connectivity_matrix[df_connectivity_matrix.index.get_level_values('subject').isin(female_con_subjects)]
    else:
        raise ValueError("Invalid combination of sex and condition")

    return df_cleaned

def pain_duration_filter(df_connectivity_matrix, category='2'):
    """
    Returns Data with only the desired patients for centrality measures. Used to measure the effect of pain duration on
    results of different graph theory metrics for comparison across visits.

    Parameters
    ----------
    df_connectivity_matrix : pandas DataFrame
        The DataFrame containing the connectivity matrix.
    category : str (default is '1')
        Categories to filter by: '1' for 4-5 months, '2' for 6-12 months, '3' for 1-4 years and '4' for 5 years and more 

    Returns
    -------
    df_connectivity_matrix : pandas DataFrame
        The filtered DataFrame.
    """
    # Check if sex is valid
    if category not in ('2', '3', '4'):
        raise ValueError("Category must be '2', '3', '4'")
    
    subjects_2 = ['sub-pl031', 'sub-pl032', 'sub-pl034', 'sub-pl035', 'sub-pl056']
    subjects_3 = ['sub-pl007', 'sub-pl010', 'sub-pl013', 'sub-pl027', 'sub-pl030', 'sub-pl039', 'sub-pl042', 'sub-pl049']
    subjects_4 = ['sub-pl012', 'sub-pl014', 'sub-pl016', 'sub-pl017', 'sub-pl019', 'sub-pl036', 'sub-pl037', 'sub-pl038', 'sub-pl047', 'sub-pl050', 'sub-pl055', 'sub-pl063', 'sub-pl064']

    # Set 'subject' as the index if it's not already
    if 'subject' not in df_connectivity_matrix.index.names:
        df_connectivity_matrix = df_connectivity_matrix.set_index('subject')
    
    elif category == '2':
        df_cleaned = df_connectivity_matrix[df_connectivity_matrix.index.get_level_values('subject').isin(subjects_2)]
    elif category == '3':
        df_cleaned = df_connectivity_matrix[df_connectivity_matrix.index.get_level_values('subject').isin(subjects_3)]
    elif category == '4':
        df_cleaned = df_connectivity_matrix[df_connectivity_matrix.index.get_level_values('subject').isin(subjects_4)]
    else:
        raise ValueError("Invalid combination")

    # Reset the index to remove the duplicated 'subject' level
    df_cleaned.index = df_cleaned.index.droplevel(0) 
    return df_cleaned

def limbic_system_filter(df_connectivity_matrix):
    df_filtered = df_connectivity_matrix.loc[df_connectivity_matrix.index.get_level_values('roi').isin([174,175,176,177,178,179,180,181,182,183,184,185,186,187,210,215,216,217,218,222,223,224])]
    df_filtered.index.names = ['subject','label', 'visit']
    df_filtered.rename(index={174: 'CG_L_7_1',175: 'CG_R_7_1',176: 'CG_L_7_2',177: 'CG_R_7_2',178: 'CG_L_7_3',179: 'CG_R_7_3',180: 'CG_L_7_4',181: 'CG_R_7_4',
                              182: 'CG_L_7_5',183: 'CG_R_7_5',184: 'CG_L_7_6',185: 'CG_R_7_6',186: 'CG_L_7_7',187: 'CG_R_7_7',210: 'Tha_L',215: 'Hipp_L',
                              216: 'Amyg_L',217: 'Accum_L',218: 'Tha_R',222: 'Hipp_R',223: 'Amyg_R',224: 'Accum_R'},
                              level='label', inplace=True)
    return df_filtered