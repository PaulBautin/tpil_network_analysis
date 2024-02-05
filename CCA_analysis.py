from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse CCA des données démopsychologiques et des métriques de la matière blanche
#
# example: python CCA_analysis.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Marc Antoine
#
# Prerequis: environnement virtuel avec python, pandas, numpy, sklearn et matplotlib (env_tpil)
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
from sklearn.datasets import load_iris
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils import shuffle
import random

def run_shuffled_cca(ddata,vdata,n_cca_components,iter):
    
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
    # Merge both Dataframes
    corr_data = pd.merge(dataset_1, dataset_2, how='inner', on=['subject', 'visit'])
    scaler = StandardScaler()
    corr_data = scaler.set_output(transform="pandas").fit_transform(corr_data)
    print(corr_data)
    # Preliminary correlations between features
    corr_coeff = corr_data.corr()
    plt.figure(figsize = (5, 5))
    sns.heatmap(corr_coeff, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
    plt.show()

def main():
    """
    main function, gather stats and call plots
    """
    # Import wm global metrics dataframe
    gtm_metrics = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global_metrics_con.csv')
    gtm_metrics = gtm_metrics.set_index(['subject', 'visit'])
    # Select only DLC patients and visit 1
    gtm_metrics_clbp_v1 = gtm_metrics.loc[(slice(None), 3), :]
    
    # Import demopsychologic results dataframe
    path_results = os.path.abspath("/mnt/c/Users/mafor/Downloads/socio_psycho_table.xlsx")
    dp_results = pd.read_excel(path_results, sheet_name="Donnee",header=0, usecols=[0,1,2,3,4,5,6])
    dp_results = dp_results.set_index(['subject', 'Groupe', 'visit'])
    # Select only DLC patients and visit 1
    dp_results_clbp_v1 = dp_results.loc[(dp_results.index.get_level_values('Groupe') == 0) & (dp_results.index.get_level_values('visit') == 3)]
    
    # # Show preliminary correlation matrix
    # correlation_matrix(gtm_metrics_clbp_v1, dp_results_clbp_v1)

    # Scale data
    scaler = StandardScaler()
    X1 = scaler.fit_transform(gtm_metrics_clbp_v1)
    X2 = scaler.fit_transform(dp_results_clbp_v1)
    
    # Choose number of canonical variates pairs (usually minimum of metrics in a dataset)
    n_comp=3

    # Test statistical significance of CCA by generating 1000 random CCAs
    rand_corr = np.array([run_shuffled_cca(X1, X2, 3, i) for i in range(1000)])[:,0]
    print(rand_corr)
    # Define CCA
    cca = CCA(scale=False, n_components=n_comp)
    
    # Transform datasets to obtain canonical variates
    X1_c, X2_c = cca.fit_transform(X1, X2)
    
    # check if there is any dependency between canonical variates (correlate canonical variate pairs)
    comp_corr = [np.corrcoef(X1_c[:, i], X2_c[:, i])[1][0] for i in range(n_comp)]
    print(comp_corr)
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=rand_corr, color='skyblue')
    plt.axhline(y=np.corrcoef(X1_c[:, 0], X2_c[:, 0])[1][0], color='red', linestyle='--', label='Actual CCA correlation')
    plt.title('Distribution of Random CCA Correlations')
    plt.xlabel('Random CCA Correlations')
    plt.ylabel('Correlation Coefficient')
    plt.legend()
    plt.show()

    # Calculate percentage of random correlations lower than or equal to actual CCA correlation
    percentage_higher = (np.sum(rand_corr <= np.corrcoef(X1_c[:, 0], X2_c[:, 0])[1][0]) / len(rand_corr)) * 100
    print(f'The actual CCA correlation is higher than {percentage_higher:.2f}% of random correlations.')
    
    # plt.bar(['CC1', 'CC2', 'CC3'], comp_corr, color='lightgrey', width=0.8, edgecolor='k')
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
    print(f'Canonical Loadings for White Matter Metrics: \n{cca.x_loadings_}') # get loadings for canonical variate of X1 dataset
    print(f'Canonical Loadings for Questionnaire results: \n{cca.y_loadings_}') # get loadings for canonical variate of X2 dataset
    
    
    # coef_df = pd.DataFrame(np.round(cca.coef_, 3), columns = [gtm_metrics_clbp_v1.columns])
    # coef_df.index = dp_results_clbp_v1.columns
    # plt.figure(figsize = (5, 5))
    # sns.heatmap(coef_df, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
    # plt.show()

if __name__ == "__main__":
    main()