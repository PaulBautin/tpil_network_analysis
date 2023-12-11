# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Fonctions pour créer figures des analyses de la connectivité structurelle
#
# example: python connectivity_analysis.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Marc Antoine
#
# Prerequis: environnement virtuel avec python, pandas, numpy et matplotlib (env_tpil)
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
from netneurotools.utils import get_centroids

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import math
import glob
from functions.connectivity_read_files import find_files_with_common_name
from netneurotools.plotting import plot_point_brain
from netneurotools.metrics import degrees_und
from functions.connectivity_filtering import load_brainnetome_centroids
from functions.connectivity_stats import mean_matrix, difference, z_score
from scipy.stats import linregress

def connectivity_matrix_viewer(conn_matrix):
    """
    Prints connectivity_matrix

    Parameters
    ----------
    conn_matrix: (N, N, S) np.array

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    conn_matrix = conn_matrix.droplevel(1)
    conn_matrix_reset = conn_matrix.reset_index()
    np_conn_matrix = np.dstack(
        list(conn_matrix_reset.groupby(['subject']).apply(lambda x: x.set_index(['subject', 'roi']).to_numpy())))
    plt.imshow(np.log(np_conn_matrix[:,:,0]), cmap='RdYlBu')
    plt.show()

def plot_network(adj, coords):
    """
    Plots `data` as a cloud of points (nodes) with edges in 3D space based on specified `coords`

    Parameters
    ----------
    adj : (N,N) array_like
        Adjacency matrix for an `N` node parcellation; determines edges to plot
    coords : (N, 3) array_like
        x, y, z coordinates for `N` node parcellation

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_facecolor('black')
    # remove nans from adj
    adj[np.isnan(adj)] = 0
    # Identify edges in the network
    edges = np.where(adj != 0)
    print(edges[0].shape)
    edge_cmap = plt.get_cmap('RdYlBu')
    #norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges].flatten()), vmax=np.max(adj[edges].flatten()))
    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    edge_val = edge_cmap(norm(adj[edges[0], edges[1]].flatten()))
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        z1, z2 = coords[edge_i, 2], coords[edge_j, 2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=c, alpha=0.5, zorder=0, linewidth=1)
    #scatter = ax.scatter(coords[:, 0],coords[:, 1],coords[:, 2], c=np.arange(1, 247), alpha=0.8)
    ax.view_init(elev=0, azim=180)
    ax.set_box_aspect((np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])))
    ax.axis('off')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('black')
    # Identify edges in the network
    # edges = np.where(adj != 0)
    # edge_cmap = plt.get_cmap('RdYlBu')
    # norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges].flatten()), vmax=np.max(adj[edges].flatten()))
    # edge_val = edge_cmap(norm(adj[edges].flatten()))
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        z1, z2 = coords[edge_i, 2], coords[edge_j, 2]
        ax2.plot([x1, x2], [y1, y2], [z1, z2], color=c, alpha=0.5, zorder=0, linewidth=1)
    # Filter coords and node_colors based on the presence of edges
    filtered_coords = coords[edges[0]]

    ax2.scatter(filtered_coords[:, 0], filtered_coords[:, 1], filtered_coords[:, 2],
                            c='gray', alpha=0.8, s=5)
    ax2.view_init(elev=90, azim=-90)
    ax2.axis('off')
    ax2.set_box_aspect((np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=edge_cmap))
    fig.tight_layout()
    plt.show()

def circle_graph(np_connectivity_matrix):
    """
    Plots a circular graph of nodes with edge intensity and brainnetome labels

    Parameters
    ----------
    np_connectivity_matrix : (N,N) array_like

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    np_connectivity_matrix[np.isnan(np_connectivity_matrix)] = 0
    A = np_connectivity_matrix.astype(float)
    N = A.shape[0] #length of matrix
    # x/y coordinates of nodes in a circular layout
    r = 1
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    # labels of nodes with Brainnetome atlas
    numbers = [f"{i+1:03d}" for i in range(N)]
    names = []
    with open('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/labels/sub-pl007_ses-v1__nativepro_seg_all_atlas.txt', 'r') as fp: 
        for line in fp:
            x = line[:-1]
            names.append(x)
    txt = [f"{name}:{number}" for name, number in zip(names, numbers)]
    # seaborn styling
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 8))
    # show nodes and edges
    plt.plot(xy[:, 0], xy[:, 1], linestyle="none", marker="o", markersize=10, color="steelblue", alpha=0.7)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] != 0:  # Considers all non-zero edge intensities
                edge_color = A[i, j]  # Use the edge intensity as the color value
                plt.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=cm.viridis(edge_color), linewidth=1)
                label_x_i = xy[i, 0] * 1.22  # Adjust label positioning for node i
                label_y_i = xy[i, 1] * 1.22  # Adjust label positioning for node i
                rotation_i = theta[i] * 180 / np.pi
                label_x_j = xy[j, 0] * 1.22  # Adjust label positioning for node j
                label_y_j = xy[j, 1] * 1.22  # Adjust label positioning for node j
                rotation_j = theta[j] * 180 / np.pi
                if theta[i] > np.pi / 2 and theta[i] < 3 * np.pi / 2:
                    rotation_i += 180  # Rotate labels on the left side of the circle by 180 degrees
                if theta[j] > np.pi / 2 and theta[j] < 3 * np.pi / 2:
                    rotation_j += 180  # Rotate labels on the left side of the circle by 180 degrees
                plt.text(label_x_i, label_y_i, txt[i], fontsize=8, rotation=rotation_i, ha='center', va='center')
                plt.text(label_x_j, label_y_j, txt[j], fontsize=8, rotation=rotation_j, ha='center', va='center')
    plt.axis([-1, 1, -1, 1])
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()

    # Create a colorbar for edge intensities
    sm = cm.ScalarMappable(cmap=cm.viridis)
    sm.set_array(A.flatten())
    cbar = plt.colorbar(sm)
    cbar.set_label('Edge Intensity')
    plt.show()

def histogram(np_connectivity_matrix):
    """
    Plots a histogram of connectivity matrix

    Parameters
    ----------
    np_connectivity_matrix : (N,N) array_like

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    data = np_connectivity_matrix.flatten()
    data[np.isnan(data)] = 0
    data_nonzero = data[data != 0]
    percentiles = np.arange(0, 100, 5) # each bin contains 5% of all data
    bin_edges = np.percentile(data_nonzero, percentiles)
    hist, bins = np.histogram(data_nonzero, bins=bin_edges)

    # Generate a list of colors for the bars using the "Pastel1" colormap
    num_bins = len(bins) - 1
    colors = plt.cm.Pastel1(np.linspace(0, 1, num_bins))

    # Plotting the histogram with colored bars
    fig, ax = plt.subplots()
    for i in range(num_bins):
        ax.bar(bins[i], hist[i], width=bins[i+1]-bins[i], color=colors[i])

    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Connectivity Matrix')

    # Add ticks to show bin edges
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{bin:.2f}' for bin in bins])

    # Display histogram
    plt.show()
    
    return hist, bins

def disruption_index(df_connectivity_con, df_connectivity_clbp):
    """
    Measures disruption of degree in individual subjects compared to control
    
    Parameters
    ----------
    df_connectivity_matrix_con : (1,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_connectivity_matrix_clbp : (1,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    
    """
    # Mean degree of control pop
    df_x = mean_matrix(df_connectivity_con)
    # Difference of degree between studied pop and control pop
    df_y = df_connectivity_clbp.groupby('subject').apply(lambda x:difference(df_x, x))
    # Merge df_x and df_y together
    df_merged = pd.merge(df_x, df_y, left_index=True, right_index=True)
    # Reset index for easier plotting
    df_merged = df_merged.droplevel(1)
    df_merged_reset = df_merged.reset_index()
    # Get unique subjects
    subjects = df_merged_reset['subject'].unique()
    num_subjects = len(subjects)
    # Dataframe to store slope values (kD)
    df_kd = pd.DataFrame(columns=['subject', 'slope'])
    # Set up subplots dynamically based on the number of subjects
    num_subjects = len(subjects)
    num_rows = math.ceil(num_subjects / 5)
    num_cols = min(num_subjects, 5)
    # Set up subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 15))
    # Iterate over subjects
    for i, subject in enumerate(subjects):
        # Specify subplot location
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        # Extract data for the current subject
        subject_data = df_merged_reset[df_merged_reset['subject'] == subject]
        x_data = subject_data['centrality_x'].to_numpy()
        y_data = subject_data['centrality_y'].to_numpy()
        # Scatter plot
        ax.scatter(x_data, y_data)
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        # Add title with subject ID
        ax.set_title(f'Subject {subject}')
        # Plot the regression line
        regression_line = intercept + slope * x_data
        ax.plot(x_data, regression_line, color='red', label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')
        # Annotate the plot with regression equation and correlation coefficient
        equation = f'R² = {r_value**2:.2f}'
        ax.annotate(equation, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10, color='red')
        # Add labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()

        # Append subject and slope values to DataFrame
        df_kd = df_kd.append({'subject': subject, 'slope': slope}, ignore_index=True)
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()
    
    return df_kd

def disruption_index_combined(df_metric_con, df_metric_clbp, df_degree_con):
    """
    Measures disruption of degree in individual subjects compared to control and plots all subjects on a single graph
    
    Parameters
    ----------
    df_connectivity_matrix_con : (1,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_connectivity_matrix_clbp : (1,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    df_kd : pandas DataFrame
        DataFrame containing subject and slope values
    """
    
    # Mean degree of controls
    df_x = mean_matrix(df_degree_con)
    
    # Mean metric of controls
    mean_metric = mean_matrix(df_metric_con)
    
    # Difference of degree between studied pop and control pop
    df_y = df_metric_clbp.groupby('subject').apply(lambda x: difference(mean_metric, x))
    
    # Merge df_x and df_y together
    df_merged = pd.merge(df_x, df_y, left_index=True, right_index=True)
    
    # Reset index for easier plotting
    df_merged = df_merged.droplevel(1)
    df_merged_reset = df_merged.reset_index()
    
    # Get unique subjects
    subjects = df_merged_reset['subject'].unique()
    
    # Dataframe to store slope values (kD)
    df_kd = pd.DataFrame(columns=['subject', 'slope'])
    
    # Set up the plot with a smaller figure size
    plt.figure(figsize=(10, 6))

    # Lists to store data for all subjects combined
    combined_x_data = []
    combined_y_data = []
    
    # Iterate over subjects
    for subject in subjects:
        # Extract data for the current subject
        subject_data = df_merged_reset[df_merged_reset['subject'] == subject]
        x_data = subject_data['centrality_x'].to_numpy()
        y_data = subject_data['centrality_y'].to_numpy()
        
        # Append data to combined lists
        combined_x_data.extend(x_data)
        combined_y_data.extend(y_data)

        # Scatter plot
        plt.scatter(x_data, y_data, label=f'Subject {subject}')
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        
        # Plot the regression line
        regression_line = intercept + slope * x_data
        plt.plot(x_data, regression_line, label=f'{subject}, y = {slope:.2f}x + {intercept:.2f}', linestyle='dashed')
        
        # Append subject and slope values to DataFrame
        df_kd = df_kd.append({'subject': subject, 'slope': slope, 'R²': r_value**2}, ignore_index=True)
    
    # Linear regression for all subjects combined
    slope_combined, intercept_combined, _, _, _ = linregress(combined_x_data, combined_y_data)
    
    # Plot the regression line for all subjects combined
    regression_line_combined = intercept_combined + slope_combined * np.array(combined_x_data)
    plt.plot(combined_x_data, regression_line_combined, label=f'Combined, y = {slope_combined:.2f}x + {intercept_combined:.2f}', linewidth=2, color='black')
    
    # Annotate the plot with labels and legend
    plt.title('Disruption of Metric of Interest in Individual Subjects Compared to Control')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Adjust legend position and font size
    
    # Show the plot
    plt.show()
    
    return df_kd
