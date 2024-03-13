from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse CCA des données démopsychologiques et des métriques de la matière blanche.
# L'analyse CCA cherche à expliquer la corrélation entre deux groupes de variables peu importe le
# nombre de variables dans chaque groupe.
#
# example: python CCA_analysis.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Marc Antoine
#
# Prerequis: environnement virtuel avec python, pandas, numpy, sklearn et matplotlib (env_tpil)
#
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
from sklearn.datasets import load_iris
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils import shuffle
import random
from functions.connectivity_processing import difference_visits
from functions.connectivity_filtering import limbic_system_filter
from functions.connectivity_stats import rank_roi

def run_shuffled_cca(ddata,vdata,n_cca_components,iter):
    """
    a simple function which runs permutations of CCA (shuffled)
    from https://github.com/CoBrALab/design-choices-cca-neuroimaging/blob/main/run_cca_function.py 
    Parameters
    ----------
    ddata : (N, M) DataFrame
    vdata : (P, Q) DataFrame
    n_cca_components : integer, number of canonical variates pairs (usually minimum of metrics in a dataset)
    iter : char, should be part of a loop to generate multiple random CCA

    Returns
    -------
    testcorrs : np.array, Pearson's correlation for every shuffled CCA
    """
    #Breaks links in dataset
    demographics_shuffled, vertexes_shuffled = shuffle(ddata, vdata, random_state=iter)
    
    #Create and run CCA model
    cca_model_shuffled = CCA(n_components=n_cca_components)
    cca_model_shuffled.fit(ddata,vertexes_shuffled)
    
    #Calculate and return the correlation coefficient
    test1_c, test2_c = cca_model_shuffled.transform(ddata, vertexes_shuffled)
    testcorrs = np.corrcoef(test1_c.T, test2_c.T).diagonal(offset=cca_model_shuffled.n_components)
    return testcorrs

def correlation_matrix(dataset_1, dataset_2):
    """
    a function to see the correlation between every variables 
    Parameters
    ----------
    ddata : (N, M) DataFrame
    vdata : (P, Q) DataFrame

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    # Merge both Dataframes
    corr_data = pd.merge(dataset_1, dataset_2, how='inner', on=['subject', 'visit'])
    scaler = StandardScaler()
    corr_data = scaler.set_output(transform="pandas").fit_transform(corr_data)
    
    # Preliminary correlations between features
    corr_coeff = corr_data.corr()
    plt.figure(figsize = (5, 5))
    sns.heatmap(corr_coeff, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
    plt.show()

def CCA_global(condition='clbp', visit=1):
    ### For global GTM
    # Import wm global metrics dataframe for CTL
    gtm_metrics_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_global_metrics_con.csv')
    gtm_metrics_con = gtm_metrics_con.set_index(['subject', 'visit', 'Unnamed: 0'])
    # Select only CTL patients and visit 1
    gtm_metrics_con_v1 = gtm_metrics_con.loc[(slice(None), visit), :]
    
    # Import wm global metrics dataframe for CLBP
    gtm_metrics_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_global_metrics.csv')
    gtm_metrics_clbp = gtm_metrics_clbp.set_index(['subject', 'visit', 'Unnamed: 0'])
    # Select only CLBP patients and visit 1
    gtm_metrics_clbp_v1 = gtm_metrics_clbp.loc[(slice(None), visit), :]
    
    # Import demopsychologic results dataframe
    path_results = os.path.abspath("/Users/Marc-Antoine/Documents/socio_psycho_table.xlsx")

    # Select CTL patients
    q_results_con = pd.read_excel(path_results, sheet_name="Donnee", header=0, usecols=[0,1,2,3,4,5,6,9])
    q_results_con = q_results_con.set_index(['subject', 'Groupe', 'visit'])
    # Select only CTL patients and visit 1
    q_results_con_v1 = q_results_con.loc[(q_results_con.index.get_level_values('Groupe') == 0) & (q_results_con.index.get_level_values('visit') == visit)]
    
    # Select CLBP patients
    q_results_clbp = pd.read_excel(path_results, sheet_name="Donnee", header=0, usecols=[0,1,2,3,4,5,6,7,8,9])
    q_results_clbp = q_results_clbp.set_index(['subject', 'Groupe', 'visit'])
    # Select only CLBP patients and visit 1
    q_results_clbp_v1 = q_results_clbp.loc[(q_results_clbp.index.get_level_values('Groupe') == 1) & (q_results_clbp.index.get_level_values('visit') == visit)]

    correlation_coefficients = {}
    canonical_loadings_x = {}
    canonical_loadings_y = {}

    # Scale data
    scaler = StandardScaler()
    if condition == 'clbp':
        X1 = scaler.fit_transform(gtm_metrics_clbp_v1)
        X2 = scaler.fit_transform(q_results_clbp_v1)
    elif condition == 'con':
        X1 = scaler.fit_transform(gtm_metrics_con_v1)
        X2 = scaler.fit_transform(q_results_con_v1)
    else:
        raise ValueError("Invalid condition")
    
    
    
    # Choose number of canonical variates pairs (usually minimum of metrics in a dataset)
    n_comp=5

    # Test statistical significance of CCA by generating 1000 random CCAs
    rand_corr = np.array([run_shuffled_cca(X1, X2, n_comp, i) for i in range(1000)])[:,0]
    
    # Define CCA
    cca = CCA(scale=False, n_components=n_comp)
    
    # Transform datasets to obtain canonical variates
    X1_c, X2_c = cca.fit_transform(X1, X2)
    
    # Calculate correlation coefficient of the canonical variates pair
    correlation_coefficient = np.corrcoef(X1_c[:, 0], X2_c[:, 0])[1][0]
    correlation_coefficients = [correlation_coefficient] # Convert ot list or array

    # Store the first column of the canonical loadings
    canonical_loadings_x = cca.x_loadings_[:, 0]
    canonical_loadings_y = cca.y_loadings_[:, 0]
    
    # Create DataFrame from the correlation coefficients
    correlation_coefficients_df = pd.DataFrame({'Correlation Coefficient': correlation_coefficients})

    # Create DataFrame from the first column of the canonical loadings
    canonical_loadings_x_df = pd.DataFrame.from_dict({'Loadings': canonical_loadings_x}, orient='index', columns=['Efficiency', 'Strength', 'Cluster', 'Small world', 'Modularity'])
    if condition == 'clbp':
        canonical_loadings_y_df = pd.DataFrame.from_dict({'Loadings': canonical_loadings_y}, orient='index', columns=['age', 'BECK', 'PCS', 'STAI', 'meanBPI', 'POQ', 'Sex'])
    elif condition == 'con':
        canonical_loadings_y_df = pd.DataFrame.from_dict({'Loadings': canonical_loadings_y}, orient='index', columns=['age', 'BECK', 'PCS', 'STAI', 'Sex'])
    else:
        raise ValueError("Invalid condition")

    # # Display results
    # print("Correlation Coefficients:")
    # print(correlation_coefficients_df)
    # print("\nCanonical Loadings:")
    # print(canonical_loadings_x_df)
    # print(canonical_loadings_y_df)
    
    canonical_loadings_x_df.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/global/loadings_x_' + str(condition) + str(visit) + '.csv')
    canonical_loadings_y_df.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/global/loadings_y_' + str(condition) + str(visit) + '.csv')
    correlation_coefficients_df.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/global/cc_' + str(condition) + str(visit) + '.csv')
    # # check if there is any dependency between canonical variates (correlate canonical variate pairs)
    # comp_corr = [np.corrcoef(X1_c[:, i], X2_c[:, i])[1][0] for i in range(n_comp)]
    # print(comp_corr)
    # plt.figure(figsize=(8, 6))
    # sns.boxplot(y=rand_corr, color='skyblue')
    # plt.axhline(y=np.corrcoef(X1_c[:, 0], X2_c[:, 0])[1][0], color='red', linestyle='--', label='Actual CCA correlation')
    # plt.title('Distribution of Random CCA Correlations')
    # plt.xlabel('Random CCA Correlations')
    # plt.ylabel('Correlation Coefficient')
    # plt.legend()
    # plt.show()

    # # Calculate percentage of random correlations lower than or equal to actual CCA correlation
    # percentage_higher = (np.sum(rand_corr <= np.corrcoef(X1_c[:, 0], X2_c[:, 0])[1][0]) / len(rand_corr)) * 100
    # print(f'The actual CCA correlation is higher than {percentage_higher:.2f}% of random correlations.')
    
    # plt.bar(['CC1', 'CC2', 'CC3', 'CC4', 'CC5'], comp_corr, color='lightgrey', width=0.8, edgecolor='k')
    # plt.xlabel('Canonical Variate Pairs')
    # plt.ylabel('Correlation Coefficient')
    # plt.title('Canonical Correlation Coefficients between Variate Pairs')
    # plt.show()

    # # Plot scatter plots for each canonical variate pair
    # for i in range(n_comp):
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(X1_c[:, i], X2_c[:, i], c='b', alpha=0.5)
    #     plt.title(f'Scatter Plot for Canonical Variate Pair {i+1}')
    #     plt.xlabel(f'Canonical Variate {i+1} from X1')
    #     plt.ylabel(f'Canonical Variate {i+1} from X2')
    #     plt.show()

    # Measure which variables influence the canonical variates the most for each dataset (use the loadings)
    # print(f'Canonical Loadings for White Matter Metrics: \n{cca.x_loadings_}') # get loadings for canonical variate of X1 dataset
    # print(f'Canonical Loadings for Questionnaire results: \n{cca.y_loadings_}') # get loadings for canonical variate of X2 dataset
    # print(f'Canonical Weights for White Matter Metrics: \n{cca.x_weights_}') # get weights for canonical variate of X1 dataset
    # print(f'Canonical Weights for Questionnaire results: \n{cca.y_weights_}') # get weights for canonical variate of X2 dataset
    
    # coef_df = pd.DataFrame(np.round(cca.coef_, 3), columns = [gtm_metrics_con_v1.columns])
    # coef_df.index = q_results_con_v1.columns
    # plt.figure(figsize = (5, 5))
    # sns.heatmap(coef_df, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
    # plt.show()

def CCA_limbic(condition='clbp', visit=1):
    ### For limbic GTM
    gtm_con = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_nodal_metrics_con.csv', index_col=['subject', 'roi', 'visit'])
    gtm_clbp = pd.read_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_nodal_metrics_clbp.csv', index_col=['subject', 'roi', 'visit'])
    gtm_limb_con = limbic_system_filter(gtm_con)
    gtm_limb_clbp = limbic_system_filter(gtm_clbp)
    
    # Select visit
    gtm_limb_con_v1 = gtm_limb_con[gtm_limb_con.index.get_level_values('visit') == visit]
    gtm_limb_clbp_v1 = gtm_limb_clbp[gtm_limb_clbp.index.get_level_values('visit') == visit]
    
    # Import demopsychologic results dataframe
    path_results = os.path.abspath("/Users/Marc-Antoine/Documents/socio_psycho_table.xlsx")

    # Select CTL patients
    q_results_con = pd.read_excel(path_results, sheet_name="Donnee", header=0, usecols=[0,1,2,3,4,5,6,9])
    q_results_con = q_results_con.set_index(['subject', 'Groupe', 'visit'])
    # Select only CTL patients and visit 1
    q_results_con_v1 = q_results_con.loc[(q_results_con.index.get_level_values('Groupe') == 0) & (q_results_con.index.get_level_values('visit') == visit)]
    
    # Select CLBP patients
    q_results_clbp = pd.read_excel(path_results, sheet_name="Donnee", header=0, usecols=[0,1,2,3,4,5,6,7,8,9])
    q_results_clbp = q_results_clbp.set_index(['subject', 'Groupe', 'visit'])
    # Select only CLBP patients and visit 1
    q_results_clbp_v1 = q_results_clbp.loc[(q_results_clbp.index.get_level_values('Groupe') == 1) & (q_results_clbp.index.get_level_values('visit') == visit)]
    
    # Show preliminary correlation matrix
    # correlation_matrix(gtm_metrics_con_v1, q_results_con_v1)

    labels = gtm_limb_clbp_v1.index.get_level_values('label').unique()
    correlation_coefficients = {}
    canonical_loadings_x = {}
    canonical_loadings_y = {}

    for label in labels:
        # Scale data
        scaler = StandardScaler()
        #Filter data for the current label
        if condition == 'clbp':
            label_data = gtm_limb_clbp_v1.loc[(slice(None), label), :]
            X1 = scaler.fit_transform(label_data)
            X2 = scaler.fit_transform(q_results_clbp_v1)
        elif condition == 'con':
            label_data = gtm_limb_con_v1.loc[(slice(None), label), :]
            X1 = scaler.fit_transform(label_data)
            X2 = scaler.fit_transform(q_results_con_v1)
        else:
            raise ValueError("Invalid condition")
        
        # Choose number of canonical variates pairs (usually minimum of metrics in a dataset)
        n_comp=5
        
        # Define CCA
        cca = CCA(scale=False, n_components=n_comp)
        
        # Transform datasets to obtain canonical variates
        X1_c, X2_c = cca.fit_transform(X1, X2)
        
        # Test statistical significance of CCA by generating 1000 random CCAs
        rand_corr = np.array([run_shuffled_cca(X1, X2, n_comp, i) for i in range(1000)])[:,0]
        # Calculate percentage of random correlations lower than or equal to actual CCA correlation
        percentage_higher = (np.sum(rand_corr <= np.corrcoef(X1_c[:, 0], X2_c[:, 0])[1][0]) / len(rand_corr)) * 100
        print(f'The actual CCA correlation is higher than {percentage_higher:.2f}% of random correlations.')

        # Calculate correlation coefficient of the canonical variates pair
        correlation_coefficient = np.corrcoef(X1_c[:, 0], X2_c[:, 0])[1][0]
        correlation_coefficients[label] = correlation_coefficient

        # Store the first column of the canonical loadings
        canonical_loadings_x[label] = cca.x_loadings_[:, 0]
        canonical_loadings_y[label] = cca.y_loadings_[:, 0]

        # # Print results for the current label
        # print(f'Correlation coefficient for label {label}: {correlation_coefficient}')
        # print(f'Canonical Loadings for White Matter Metrics for label {label}: \n{cca.x_loadings_[:, 0]}')
        # print(f'Canonical Loadings for Questionnaire results for label {label}: \n{cca.y_loadings_[:, 0]}')
        # print('\n')

    # Create DataFrame from the correlation coefficients
    correlation_coefficients_df = pd.DataFrame(list(correlation_coefficients.items()), columns=['Label', 'Correlation Coefficient'])

    # Create DataFrame from the first column of the canonical loadings
    canonical_loadings_x_df = pd.DataFrame.from_dict(canonical_loadings_x, orient='index', columns=['Degree', 'Strength', 'Betweenness', 'Cluster', 'Efficiency'])
    if condition == 'clbp':
        canonical_loadings_y_df = pd.DataFrame.from_dict(canonical_loadings_y, orient='index', columns=['age', 'BECK', 'PCS', 'STAI', 'meanBPI', 'POQ', 'Sex'])
    elif condition == 'con':
        canonical_loadings_y_df = pd.DataFrame.from_dict(canonical_loadings_y, orient='index', columns=['age', 'BECK', 'PCS', 'STAI', 'Sex'])
    else:
        raise ValueError("Invalid condition")
    # # Display results
    # print("Correlation Coefficients:")
    # print(correlation_coefficients_df)
    # print("\nCanonical Loadings:")
    # print(canonical_loadings_x_df)
    # print(canonical_loadings_y_df)
    
    canonical_loadings_x_df.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/limbic/loadings_x_' + str(condition) + str(visit) + '.csv')
    canonical_loadings_y_df.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/limbic/loadings_y_' + str(condition) + str(visit) + '.csv')
    correlation_coefficients_df.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/cca/limbic/cc_' + str(condition) + str(visit) + '.csv')

def main():
    """
    main function, gather stats and call plots
    """
    
    CCA_limbic(condition='clbp', visit=1)
    CCA_limbic(condition='clbp', visit=2)
    CCA_limbic(condition='clbp', visit=3)
    CCA_limbic(condition='con', visit=1)
    CCA_limbic(condition='con', visit=2)
    CCA_limbic(condition='con', visit=3)
    
    



if __name__ == "__main__":
    main()