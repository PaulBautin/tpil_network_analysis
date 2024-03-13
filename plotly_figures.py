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
import plotly.express as px
import glob
import bct
import networkx as nx
import pingouin as pg
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import t
from statsmodels.stats.anova import AnovaRM
from functions.connectivity_figures import plot_network, circle_graph, histogram, connectivity_matrix_viewer
from functions.connectivity_filtering import distance_dependant_filter, load_brainnetome_centroids, scilpy_filter, threshold_filter, sex_filter, pain_duration_filter, limbic_system_filter
from functions.connectivity_processing import data_processor, prepare_data
from functions.connectivity_read_files import find_files_with_common_name
from functions.gtm_bct import  compute_betweenness, compute_cluster, compute_degree, compute_eigenvector, compute_efficiency, compute_small_world, compute_shortest_path, modularity_louvain, bct_master
from functions.connectivity_stats import mean_matrix, z_score, friedman, nbs_data, my_icc, icc, calculate_icc_all_rois, calculate_cv
from functions.gtm_nx import networkx_graph_convertor

def friedman_figure():
    """
    Graph Friedman test for GTM on local 
    """
    loc_deg_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/degree_clbp.csv')
    loc_deg_clbp = loc_deg_clbp['statistic'].reset_index()
    loc_deg_clbp['Condition'] = 1
    loc_deg_clbp['Metric'] = 1
    loc_str_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/strength_clbp.csv')
    loc_str_clbp = loc_str_clbp['statistic'].reset_index()
    loc_str_clbp['Condition'] = 1
    loc_str_clbp['Metric'] = 2
    loc_bet_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/betweenness_clbp.csv')
    loc_bet_clbp = loc_bet_clbp['statistic'].reset_index()
    loc_bet_clbp['Condition'] = 1
    loc_bet_clbp['Metric'] = 3
    loc_clu_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/cluster_clbp.csv')
    loc_clu_clbp = loc_clu_clbp['statistic'].reset_index()
    loc_clu_clbp['Condition'] = 1
    loc_clu_clbp['Metric'] = 4
    loc_eff_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/efficiency_clbp.csv')
    loc_eff_clbp = loc_eff_clbp['statistic'].reset_index()
    loc_eff_clbp['Condition'] = 1
    loc_eff_clbp['Metric'] = 5

    loc_deg_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/degree_con.csv')
    loc_deg_con = loc_deg_con['statistic'].reset_index()
    loc_deg_con['Condition'] = 2
    loc_deg_con['Metric'] = 1.2
    loc_str_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/strength_con.csv')
    loc_str_con = loc_str_con['statistic'].reset_index()
    loc_str_con['Condition'] = 2
    loc_str_con['Metric'] = 2.2
    loc_bet_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/betweenness_con.csv')
    loc_bet_con = loc_bet_con['statistic'].reset_index()
    loc_bet_con['Condition'] = 2
    loc_bet_con['Metric'] = 3.2
    loc_clu_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/cluster_con.csv')
    loc_clu_con = loc_clu_con['statistic'].reset_index()
    loc_clu_con['Condition'] = 2
    loc_clu_con['Metric'] = 4.2
    loc_eff_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/nodal/efficiency_con.csv')
    loc_eff_con = loc_eff_con['statistic'].reset_index()
    loc_eff_con['Condition'] = 2
    loc_eff_con['Metric'] = 5.2

    # df =pd.concat([loc_deg_clbp,loc_deg_con,loc_str_clbp,loc_str_con,loc_bet_clbp,loc_bet_con,loc_clu_clbp,loc_clu_con,loc_eff_clbp,loc_eff_con])
    # df['Metric'] = df['Metric'].replace({1: 'degree centrality', 2: 'strength centrality', 3: 'betweenness centrality', 4: 'cluster coefficient', 5:'efficiency'})
    # df['Condition'] = df['Condition'].replace({1: 'CLBP', 2: 'CTL'})
    # df.rename(columns={'statistic': 'X<sup>2</sup> statistic'}, inplace=True)
    # fig = px.scatter_3d(df, x='Metric', y='Condition', z='X<sup>2</sup> statistic', size_max=18,
    #                     color='Condition', color_discrete_map={'CLBP': 'red', 'CTL': 'blue'})
    # # Update layout to adjust spacing between categories and ticks on z-axis
    # fig.update_layout(scene=dict(zaxis=dict(tickmode='linear', tick0=0, dtick=1)))
    # fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=0.50, z=1)))
    # # Add a line to highlight significance at z=6
    # fig.add_scatter3d(x=['degree centrality', 'strength centrality', 'betweenness centrality', 'cluster coefficient', 'efficiency'], 
    #                 y=['CLBP', 'CLBP', 'CLBP', 'CLBP', 'CLBP', 'CLBP'], 
    #                 z=[6, 6, 6, 6, 6, 6],
    #                 mode='lines',
    #                 opacity=0.5, 
    #                 line=dict(color='black', width=3))
    # fig.add_scatter3d(x=['degree centrality', 'strength centrality', 'betweenness centrality', 'cluster coefficient', 'efficiency'], 
    #                 y=['CTL', 'CTL', 'CTL', 'CTL', 'CTL', 'CTL'], 
    #                 z=[6, 6, 6, 6, 6, 6],
    #                 mode='lines',
    #                 opacity=0.5, 
    #                 line=dict(color='black', width=3))
    # fig.show()
    
    """
    Graph Friedman test for GTM on limbic system
    """
    limb_deg_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/degree_clbp.csv')
    limb_deg_clbp = limb_deg_clbp['statistic'].reset_index()
    limb_deg_clbp['Condition'] = 1
    limb_deg_clbp['Metric'] = 6
    limb_str_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/strength_clbp.csv')
    limb_str_clbp = limb_str_clbp['statistic'].reset_index()
    limb_str_clbp['Condition'] = 1
    limb_str_clbp['Metric'] = 7
    limb_bet_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/betweenness_clbp.csv')
    limb_bet_clbp = limb_bet_clbp['statistic'].reset_index()
    limb_bet_clbp['Condition'] = 1
    limb_bet_clbp['Metric'] = 8
    limb_clu_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/cluster_clbp.csv')
    limb_clu_clbp = limb_clu_clbp['statistic'].reset_index()
    limb_clu_clbp['Condition'] = 1
    limb_clu_clbp['Metric'] = 9
    limb_eff_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/efficiency_clbp.csv')
    limb_eff_clbp = limb_eff_clbp['statistic'].reset_index()
    limb_eff_clbp['Condition'] = 1
    limb_eff_clbp['Metric'] = 10

    limb_deg_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/degree_con.csv')
    limb_deg_con = limb_deg_con['statistic'].reset_index()
    limb_deg_con['Condition'] = 2
    limb_deg_con['Metric'] = 6.2
    limb_str_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/strength_con.csv')
    limb_str_con = limb_str_con['statistic'].reset_index()
    limb_str_con['Condition'] = 2
    limb_str_con['Metric'] = 7.2
    limb_bet_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/betweenness_con.csv')
    limb_bet_con = limb_bet_con['statistic'].reset_index()
    limb_bet_con['Condition'] = 2
    limb_bet_con['Metric'] = 8.2
    limb_clu_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/cluster_con.csv')
    limb_clu_con = limb_clu_con['statistic'].reset_index()
    limb_clu_con['Condition'] = 2
    limb_clu_con['Metric'] = 9.2
    limb_eff_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/limbic/efficiency_con.csv')
    limb_eff_con = limb_eff_con['statistic'].reset_index()
    limb_eff_con['Condition'] = 2
    limb_eff_con['Metric'] = 10.2

    # df =pd.concat([limb_deg_clbp,limb_deg_con,limb_str_clbp,limb_str_con,limb_bet_clbp,limb_bet_con,limb_clu_clbp,limb_clu_con,limb_eff_clbp,limb_eff_con])
    # df['Metric'] = df['Metric'].replace({6: 'degree centrality', 7: 'strength centrality', 8: 'betweenness centrality', 9: 'cluster coefficient', 10:'efficiency'})
    # df['condition'] = df['condition'].replace({1: 'CLBP', 2: 'CTL'})
    # df.rename(columns={'statistic': 'X<sup>2</sup> statistic'}, inplace=True)
    # fig = px.scatter_3d(df, x='Metric', y='Condition', z='X<sup>2</sup> statistic', size_max=18,
    #                     opacity=1.0, color='Condition', color_discrete_map={'CLBP': 'red', 'CTL': 'blue'})
    # # Update layout to adjust spacing between categories and ticks on z-axis
    # fig.update_layout(scene=dict(zaxis=dict(tickmode='linear', tick0=0, dtick=1)))
    # fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=0.50, z=1)))
    # # Add a line to highlight significance at z=6
    # fig.add_scatter3d(x=['degree centrality', 'strength centrality', 'betweenness centrality', 'cluster coefficient', 'efficiency'], 
    #                 y=['CLBP', 'CLBP', 'CLBP', 'CLBP', 'CLBP', 'CLBP'], 
    #                 z=[6, 6, 6, 6, 6, 6],
    #                 mode='lines',
    #                 opacity=0.5, 
    #                 line=dict(color='black', width=3))
    # fig.add_scatter3d(x=['degree centrality', 'strength centrality', 'betweenness centrality', 'cluster coefficient', 'efficiency'], 
    #                 y=['CTL', 'CTL', 'CTL', 'CTL', 'CTL', 'CTL'], 
    #                 z=[6, 6, 6, 6, 6, 6],
    #                 mode='lines',
    #                 opacity=0.5, 
    #                 line=dict(color='black', width=3))
    # fig.show()

    """
    Graph Friedman test for GTM on global 
    """
    glo_sw_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/small_world_clbp.csv')
    glo_sw_clbp = glo_sw_clbp['statistic'].reset_index()
    glo_sw_clbp['Condition'] = 1
    glo_sw_clbp['Metric'] = 11
    glo_str_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/strength_clbp.csv')
    glo_str_clbp = glo_str_clbp['statistic'].reset_index()
    glo_str_clbp['Condition'] = 1
    glo_str_clbp['Metric'] = 12
    glo_mod_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/modularity_clbp.csv')
    glo_mod_clbp = glo_mod_clbp['statistic'].reset_index()
    glo_mod_clbp['Condition'] = 1
    glo_mod_clbp['Metric'] = 13
    glo_clu_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/cluster_clbp.csv')
    glo_clu_clbp = glo_clu_clbp['statistic'].reset_index()
    glo_clu_clbp['Condition'] = 1
    glo_clu_clbp['Metric'] = 14
    glo_eff_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/efficiency_clbp.csv')
    glo_eff_clbp = glo_eff_clbp['statistic'].reset_index()
    glo_eff_clbp['Condition'] = 1
    glo_eff_clbp['Metric'] = 15

    glo_sw_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/small_world_con.csv')
    glo_sw_con = glo_sw_con['statistic'].reset_index()
    glo_sw_con['Condition'] = 2
    glo_sw_con['Metric'] = 11.2
    glo_str_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/strength_con.csv')
    glo_str_con = glo_str_con['statistic'].reset_index()
    glo_str_con['Condition'] = 2
    glo_str_con['Metric'] = 12.2
    glo_mod_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/modularity_con.csv')
    glo_mod_con = glo_mod_con['statistic'].reset_index()
    glo_mod_con['Condition'] = 2
    glo_mod_con['Metric'] = 13.2
    glo_clu_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/cluster_con.csv')
    glo_clu_con = glo_clu_con['statistic'].reset_index()
    glo_clu_con['Condition'] = 2
    glo_clu_con['Metric'] = 14.2
    glo_eff_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/friedman/global/efficiency_con.csv')
    glo_eff_con = glo_eff_con['statistic'].reset_index()
    glo_eff_con['Condition'] = 2
    glo_eff_con['Metric'] = 15.2

    # df =pd.concat([glo_sw_clbp,glo_sw_con,glo_str_clbp,glo_str_con,glo_mod_clbp,glo_mod_con,glo_clu_clbp,glo_clu_con,glo_eff_clbp,glo_eff_con])
    # df['Metric'] = df['Metric'].replace({11: 'small worldness', 12: 'global strength', 13: 'modularity', 14: 'global cluster coefficient', 15:' global efficiency'})
    # df['Condition'] = df['Condition'].replace({1: 'CLBP', 2: 'CTL'})
    # df.rename(columns={'statistic': 'X<sup>2</sup> statistic'}, inplace=True)
    # fig = px.scatter_3d(df, x='Metric', y='Condition', z='X<sup>2</sup> statistic', size_max=18,
    #                     opacity=1.0, color='Condition', color_discrete_map={'CLBP': 'red', 'CTL': 'blue'})
    # # Update layout to adjust spacing between categories and ticks on z-axis
    # fig.update_layout(scene=dict(zaxis=dict(tickmode='linear', tick0=0, dtick=1)))
    # # Update layout to adjust aspect ratio
    # fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=0.75, y=0.33, z=1)))
    # # Add a line to highlight significance at z=6
    # fig.add_scatter3d(x=['small worldness', 'global strength', 'modularity', 'global cluster coefficient', 'global efficiency'], 
    #                 y=['CLBP', 'CLBP', 'CLBP', 'CLBP', 'CLBP', 'CLBP'], 
    #                 z=[6, 6, 6, 6, 6, 6],
    #                 mode='lines',
    #                 opacity=0.5, 
    #                 line=dict(color='black', width=3))
    # fig.add_scatter3d(x=['small worldness', 'global strength', 'modularity', 'global cluster coefficient', 'global efficiency'], 
    #                 y=['CTL', 'CTL', 'CTL', 'CTL', 'CTL', 'CTL'], 
    #                 z=[6, 6, 6, 6, 6, 6],
    #                 mode='lines',
    #                 opacity=0.5, 
    #                 line=dict(color='black', width=3))
    # fig.show()

    """
    Graph all together
    """
    df =pd.concat([loc_deg_clbp,loc_deg_con,loc_str_clbp,loc_str_con,loc_bet_clbp,loc_bet_con,loc_clu_clbp,loc_clu_con,loc_eff_clbp,loc_eff_con,
                   limb_deg_clbp,limb_deg_con,limb_str_clbp,limb_str_con,limb_bet_clbp,limb_bet_con,limb_clu_clbp,limb_clu_con,limb_eff_clbp,limb_eff_con,
                   glo_sw_clbp,glo_sw_con,glo_str_clbp,glo_str_con,glo_mod_clbp,glo_mod_con,glo_clu_clbp,glo_clu_con,glo_eff_clbp,glo_eff_con])
    # df['Metric'] = df['Metric'].replace({1: 'DC local', 2: 'SC local', 3: 'BC local', 4: 'CC local', 5:'E local',
    #                                      6: 'DC limbic', 7: 'SC limbic', 8: 'BC limbic', 9: 'CC limbic', 10:'E limbic',
    #                                      11: 'SW global', 12: 'SC global', 13: 'M global', 14: 'CC global', 15:'E global'})
    df['Condition'] = df['Condition'].replace({1: 'CLBP', 2: 'CTL'})
    df.rename(columns={'statistic': 'X<sup>2</sup> statistic'}, inplace=True)
    # Create a scatter plot
    fig = px.scatter(df, x='Metric', y='X<sup>2</sup> statistic', size_max=18,
                        color='Condition', color_discrete_map={'CLBP': 'red', 'CTL': 'blue'})
    # Update layout to adjust spacing between categories and ticks
    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1.5, y=0.50)))
    # Update the x-axis tick angle for better readability
    fig.update_layout(xaxis_tickangle=-45)
    # Increase the size of the points
    fig.update_traces(marker=dict(size=18))
    # Update the font size of the axes titles and tick labels
    fig.update_layout(
        xaxis=dict(title='Metric', tickfont=dict(size=20), title_font=dict(size=24)),
        yaxis=dict(title='X<sup>2</sup> statistic', tickfont=dict(size=20), title_font=dict(size=24)),
        )
    # Update tick labels of x-axis
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1,13.1,14.1,15.1],
            ticktext = ['DC local', 'SC local', 'BC local', 'CC local', 'E local',
                         'DC limbic', 'SC limbic', 'BC limbic', 'CC limbic', 'E limbic',
                         'SW global', 'SC global', 'M global', 'CC global', 'E global']
        )
    )
    # Add a line to highlight significance at z=6
    fig.add_scatter(x=[0,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1,13.1,14.1,15.1,16.1], 
                    y=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,6],
                    mode='lines',
                    opacity=0.5, 
                    line=dict(color='black', width=3))
    
    fig.show()

def canonical_corr_fig(level='limbic'):
    # Call canonical correlation coefficients
    limb_cc_clbp1 = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/cc_clbp1.csv')
    limb_cc_clbp1 = limb_cc_clbp1['Correlation Coefficient'].reset_index()
    limb_cc_clbp1['Condition'] = 1
    limb_cc_clbp1['Visit'] = 1
    limb_cc_clbp1['Metric'] = 1.0
    limb_cc_clbp2 = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/cc_clbp2.csv')
    limb_cc_clbp2 = limb_cc_clbp2['Correlation Coefficient'].reset_index()
    limb_cc_clbp2['Condition'] = 1
    limb_cc_clbp2['Visit'] = 2
    limb_cc_clbp2['Metric'] = 1.2
    limb_cc_clbp3 = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/cc_clbp3.csv')
    limb_cc_clbp3 = limb_cc_clbp3['Correlation Coefficient'].reset_index()
    limb_cc_clbp3['Condition'] = 1
    limb_cc_clbp3['Visit'] = 3
    limb_cc_clbp3['Metric'] = 1.4
    limb_cc_con1 = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/cc_con1.csv')
    limb_cc_con1 = limb_cc_con1['Correlation Coefficient'].reset_index()
    limb_cc_con1['Condition'] = 2
    limb_cc_con1['Visit'] = 1
    limb_cc_con1['Metric'] = 2.0
    limb_cc_con2 = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/cc_con2.csv')
    limb_cc_con2 = limb_cc_con2['Correlation Coefficient'].reset_index()
    limb_cc_con2['Condition'] = 2
    limb_cc_con2['Visit'] = 2
    limb_cc_con2['Metric'] = 2.2
    limb_cc_con3 = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/cc_con3.csv')
    limb_cc_con3 = limb_cc_con3['Correlation Coefficient'].reset_index()
    limb_cc_con3['Condition'] = 2
    limb_cc_con3['Visit'] = 3
    limb_cc_con3['Metric'] = 2.4

    """
    Graph all together
    """
    df =pd.concat([limb_cc_clbp1,limb_cc_clbp2,limb_cc_clbp3,limb_cc_con1,limb_cc_con2,limb_cc_con3])
    df['Condition'] = df['Condition'].replace({1: 'CLBP', 2: 'CTL'})
    
    # Create a scatter plot
    fig = px.scatter(df, x='Metric', y='Correlation Coefficient', size_max=50,
                        symbol='Visit', color='Condition', color_discrete_map={'CLBP': 'red', 'CTL': 'blue'})
    # Update layout to adjust spacing between categories and ticks
    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1.5, y=0.33)))
    # Update the font size of the axes titles and tick labels
    fig.update_layout(
        xaxis=dict(title='Metric', tickfont=dict(size=30), title_font=dict(size=24)),
        yaxis=dict(title='Correlation Coefficient', tickfont=dict(size=30), title_font=dict(size=24)),
        )
    # Increase the size of the points
    fig.update_traces(marker=dict(size=22))
    # Update tick labels of x-axis
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [1.2, 2.2, 3.6, 4.6],
            ticktext = ['CLBP nodal', 'CTL nodal', 'CLBP limbic', 'CTL limbic']
        )
    )
    
    fig.show()

def canonical_load_fig(level='limbic'):
    # Call canonical loadings x
    limb_deg_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp1.csv')['Degree'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 1.0})
    limb_deg_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp2.csv')['Degree'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 1.1})
    limb_deg_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp3.csv')['Degree'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 1.2})
    limb_deg_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con1.csv')['Degree'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 1.4})
    limb_deg_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con2.csv')['Degree'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 1.5})
    limb_deg_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con3.csv')['Degree'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 1.6})
    limb_str_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp1.csv')['Strength'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 2.0})
    limb_str_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp2.csv')['Strength'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 2.1})
    limb_str_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp3.csv')['Strength'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 2.2})
    limb_str_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con1.csv')['Strength'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 2.4})
    limb_str_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con2.csv')['Strength'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 2.5})
    limb_str_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con3.csv')['Strength'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 2.6})
    limb_bet_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp1.csv')['Betweenness'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 3.0})
    limb_bet_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp2.csv')['Betweenness'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 3.1})
    limb_bet_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp3.csv')['Betweenness'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 3.2})
    limb_bet_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con1.csv')['Betweenness'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 3.4})
    limb_bet_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con2.csv')['Betweenness'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 3.5})
    limb_bet_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con3.csv')['Betweenness'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 3.6})
    limb_clu_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp1.csv')['Cluster'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 4.0})
    limb_clu_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp2.csv')['Cluster'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 4.1})
    limb_clu_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp3.csv')['Cluster'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 4.2})
    limb_clu_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con1.csv')['Cluster'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 4.4})
    limb_clu_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con2.csv')['Cluster'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 4.5})
    limb_clu_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con3.csv')['Cluster'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 4.6})
    limb_eff_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp1.csv')['Efficiency'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 5.0})
    limb_eff_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp2.csv')['Efficiency'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 5.1})
    limb_eff_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_clbp3.csv')['Efficiency'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 5.2})
    limb_eff_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con1.csv')['Efficiency'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 5.4})
    limb_eff_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con2.csv')['Efficiency'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 5.5})
    limb_eff_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_x_con3.csv')['Efficiency'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 5.6})
    # Call canonical loadings y
    limb_age_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp1.csv')['Age'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 6.0})
    limb_age_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp2.csv')['Age'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 6.1})
    limb_age_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp3.csv')['Age'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 6.2})
    limb_age_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con1.csv')['Age'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 6.4})
    limb_age_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con2.csv')['Age'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 6.5})
    limb_age_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con3.csv')['Age'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 6.6})
    limb_beck_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp1.csv')['BECK'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 7.0})
    limb_beck_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp2.csv')['BECK'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 7.1})
    limb_beck_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp3.csv')['BECK'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 7.2})
    limb_beck_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con1.csv')['BECK'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 7.4})
    limb_beck_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con2.csv')['BECK'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 7.5})
    limb_beck_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con3.csv')['BECK'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 7.6})
    limb_pcs_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp1.csv')['PCS'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 8.0})
    limb_pcs_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp2.csv')['PCS'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 8.1})
    limb_pcs_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp3.csv')['PCS'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 8.2})
    limb_pcs_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con1.csv')['PCS'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 8.4})
    limb_pcs_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con2.csv')['PCS'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 8.5})
    limb_pcs_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con3.csv')['PCS'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 8.6})
    limb_stai_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp1.csv')['STAI'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 9.0})
    limb_stai_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp2.csv')['STAI'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 9.1})
    limb_stai_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp3.csv')['STAI'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 9.2})
    limb_stai_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con1.csv')['STAI'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 9.4})
    limb_stai_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con2.csv')['STAI'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 9.5})
    limb_stai_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con3.csv')['STAI'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 9.6})
    limb_sex_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp1.csv')['Sex'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 10.0})
    limb_sex_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp2.csv')['Sex'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 10.1})
    limb_sex_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp3.csv')['Sex'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 10.2})
    limb_sex_con1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con1.csv')['Sex'],
                                 'Condition': 2,
                                 'Visit': 1,
                                 'Metric': 10.4})
    limb_sex_con2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con2.csv')['Sex'],
                                 'Condition': 2,
                                 'Visit': 2,
                                 'Metric': 10.5})
    limb_sex_con3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_con3.csv')['Sex'],
                                 'Condition': 2,
                                 'Visit': 3,
                                 'Metric': 10.6})
    limb_bpi_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp1.csv')['meanBPI'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 11.0})
    limb_bpi_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp2.csv')['meanBPI'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 11.1})
    limb_bpi_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp3.csv')['meanBPI'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 11.2})
    limb_poq_clbp1 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp1.csv')['POQ'],
                                 'Condition': 1,
                                 'Visit': 1,
                                 'Metric': 12.0})
    limb_poq_clbp2 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp2.csv')['POQ'],
                                 'Condition': 1,
                                 'Visit': 2,
                                 'Metric': 12.1})
    limb_poq_clbp3 = pd.DataFrame({'Loadings': pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/' + str(level) + '/loadings_y_clbp3.csv')['POQ'],
                                 'Condition': 1,
                                 'Visit': 3,
                                 'Metric': 12.2})


    """
    Graph all together
    """
    df =pd.concat([limb_deg_clbp1, limb_deg_clbp2, limb_deg_clbp3, limb_deg_con1, limb_deg_con2, limb_deg_con3,
                   limb_str_clbp1, limb_str_clbp2, limb_str_clbp3, limb_str_con1, limb_str_con2, limb_str_con3,
                   limb_bet_clbp1, limb_bet_clbp2, limb_bet_clbp3, limb_bet_con1, limb_bet_con2, limb_bet_con3,
                   limb_clu_clbp1, limb_clu_clbp2, limb_clu_clbp3, limb_clu_con1, limb_clu_con2, limb_clu_con3,
                   limb_eff_clbp1, limb_eff_clbp2, limb_eff_clbp3, limb_eff_con1, limb_eff_con2, limb_eff_con3,
                   limb_age_clbp1, limb_age_clbp2, limb_age_clbp3, limb_age_con1, limb_age_con2, limb_age_con3,
                   limb_sex_clbp1, limb_sex_clbp2, limb_sex_clbp3, limb_sex_con1, limb_sex_con2, limb_sex_con3,
                   limb_beck_clbp1, limb_beck_clbp2, limb_beck_clbp3, limb_beck_con1, limb_beck_con2, limb_beck_con3,
                   limb_pcs_clbp1,limb_pcs_clbp2, limb_pcs_clbp3, limb_pcs_con1, limb_pcs_con2, limb_pcs_con3,
                   limb_stai_clbp1, limb_stai_clbp2, limb_stai_clbp3, limb_stai_con1, limb_stai_con2, limb_stai_con3,
                   limb_bpi_clbp1, limb_bpi_clbp2, limb_bpi_clbp3, limb_poq_clbp1, limb_poq_clbp2, limb_poq_clbp3], ignore_index=False)
    df['Condition'] = df['Condition'].replace({1: 'CLBP', 2: 'CTL'})
    print(df)
    # Create a scatter plot
    fig = px.scatter(df, x='Metric', y='Loadings', size_max=50,
                        symbol='Visit', color='Condition', color_discrete_map={'CLBP': 'red', 'CTL': 'blue'})
    # Update layout to adjust spacing between categories and ticks
    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1.5, y=0.33)))
    # Update the font size of the axes titles and tick labels
    fig.update_layout(
        xaxis=dict(title='Metric', tickfont=dict(size=30), title_font=dict(size=24)),
        yaxis=dict(title='Canonical loadings', tickfont=dict(size=30), title_font=dict(size=24)),
        )
    
    # Increase the size of the points
    fig.update_traces(marker=dict(size=22))
    # Update tick labels of x-axis
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 12.3],
            ticktext = ['Degree', 'Strength', 'Betweenness', 'Cluster', 'Efficiency', 
                        'Age', 'Sex', 'BECK', 'PCS', 'STAI', 'mean BPI', 'POQ']
        )
    )
    
    fig.show()


def main():
    """
    main function, gather stats and call plots
    """
    ### Friedman figure
    friedman_figure()

    ### Canonical correlation figure
    canonical_corr_fig(level='limbic')
    canonical_load_fig(level='limbic')

if __name__ == "__main__":
    main()