conda activate env_scil

"""
This will result in a binary mask where 90% of the population has at least 1 streamline
and 90% of the population has at least 20mm of average streamlines length.
"""
# each node with a value of 1 represents a node with at least 90% of the population having at least 1 streamline
scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v1_mask_sc.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/clbp/*_ses-v1/Compute_Connectivity/sc.npy 1 0.90 -f -v

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v2_mask_sc.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/clbp/*_ses-v2/Compute_Connectivity/sc.npy 1 0.90 -f -v

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v3_mask_sc.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/clbp/*_ses-v3/Compute_Connectivity/sc.npy 1 0.90 -f -v

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v1_mask_sc.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/control/*_ses-v1/Compute_Connectivity/sc.npy 1 0.90 -f -v

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v2_mask_sc.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/control/*_ses-v2/Compute_Connectivity/sc.npy 1 0.90 -f -v

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v3_mask_sc.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/control/*_ses-v3/Compute_Connectivity/sc.npy 1 0.90 -f -v

# each node with a value of 1 represents a node with at least 90% of the population having at least 20mm of average streamlines length

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v1_mask_len.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/clbp/*_ses-v1/Compute_Connectivity/len.npy 10 0.90 -v -f

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v2_mask_len.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/clbp/*_ses-v2/Compute_Connectivity/len.npy 10 0.90 -v -f

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v3_mask_len.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/clbp/*_ses-v3/Compute_Connectivity/len.npy 10 0.90 -v -f

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v1_mask_len.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/control/*_ses-v1/Compute_Connectivity/len.npy 10 0.90 -v -f

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v2_mask_len.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/control/*_ses-v2/Compute_Connectivity/len.npy 10 0.90 -v -f

scil_filter_connectivity.py /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v3_mask_len.npy \
    --greater_than /home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/data/23-08-21_brainnetome/control/*_ses-v3/Compute_Connectivity/len.npy 10 0.90 -v -f