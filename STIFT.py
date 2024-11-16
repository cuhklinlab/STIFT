import os
import sys
sys.path.append('/project/Stat/s1155202253/myproject/DeST_OT/src/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional
import scanpy as sc
import scipy.sparse as sp
import sklearn.neighbors
import networkx as nx
import scipy

import anndata as ad
import scipy.linalg

import STAligner
from STIFT.train_STIFT import train_STIFT
from destot.DESTOT import align

def get_mapping(Pi):
    mapping_AB = np.argmax(Pi, axis=1)
    mapping_BA = np.argmax(Pi, axis=0)
    return mapping_AB, mapping_BA

def get_topk_mapping(Pi, k=1):
    """
    Get the top k argmax values by row and by column from a matrix Pi.

    Args:
        Pi (numpy.ndarray): Input matrix.
        k (int): Number of top values to return.

    Returns:
        top_k_row_indices (numpy.ndarray): Array where each row contains the top k indices by row.
        top_k_col_indices (numpy.ndarray): Array where each column contains the top k indices by column.
    """
    row_argmax = np.argpartition(-Pi, k, axis=1)[:, :k]
    col_argmax = np.argpartition(-Pi, k, axis=0)[:k, :]

    return row_argmax, col_argmax

def preprocess_adata_list(adata_list, section_ids, k_cutoff=10, n_top_genes=5000):
    """
    Process a list of AnnData objects for different sections.

    Parameters:
    adata_list (list): List of AnnData objects
    section_ids (list): List of section IDs corresponding to the AnnData objects
    k_cutoff (int): k value for KNN in spatial network calculation (default: 10)
    n_top_genes (int): Number of top highly variable genes to select (default: 5000)

    Returns:
    tuple: (Batch_list, adj_list) where Batch_list is a list of processed AnnData objects
           and adj_list is a list of adjacency matrices
    """
    Batch_list = []
    adj_list = []

    for section_id, adata in zip(section_ids, adata_list):
        print(f"Processing section: {section_id}")

        # Uncomment the following line if you want to make spot names unique
        # adata.obs_names = [f"{x}_{section_id}" for x in adata.obs_names]

        # Calculate spatial network
        STAligner.Cal_Spatial_Net(adata, model="KNN", k_cutoff=k_cutoff)
        
        # Normalization
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Identify highly variable genes
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)

        # Subset to highly variable genes
        adata = adata[:, adata.var['highly_variable']]

        # Append results to lists
        adj_list.append(adata.uns['adj'])
        Batch_list.append(adata)

    return adata_list, Batch_list, adj_list

def get_family_information_from_downsample(downsampled_adata_list, adata_list, downsampled_Pi_list, section_ids, k=3):
    topk_mapping_AB_list = []
    topk_mapping_BA_list = []

    # Extract top-k mappings from the downsampled Pi matrices
    for Pi in downsampled_Pi_list:
        topk_mapping_AB, topk_mapping_BA = get_topk_mapping(Pi, k=k)
        topk_mapping_AB_list.append(topk_mapping_AB)
        topk_mapping_BA_list.append(topk_mapping_BA)

    for i in range(len(section_ids)-1):
        downsampled_adata_list[i].obsm["children"] = topk_mapping_AB_list[i]
        downsampled_adata_list[i+1].obsm["parents"] = topk_mapping_BA_list[i].T

    for i in range(len(adata_list) - 1):
        # Create children dictionary for downsampled adata_list[i]
        children_dict = {}
        children = downsampled_adata_list[i].obsm["children"]
        for cell_idx, child_indices in enumerate(children):
            valid_children = [idx for idx in child_indices if idx != -1]  # Assuming -1 represents no child
            if valid_children:
                cell_name = downsampled_adata_list[i].obs_names[cell_idx]
                child_names = downsampled_adata_list[i+1].obs_names[valid_children].tolist()
                children_dict[cell_name] = child_names

        # Create parents dictionary for downsampled adata_list[i+1]
        parents_dict = {}
        parents = downsampled_adata_list[i+1].obsm["parents"]
        for cell_idx, parent_indices in enumerate(parents):
            valid_parents = [idx for idx in parent_indices if idx != -1]  # Assuming -1 represents no parent
            if valid_parents:
                cell_name = downsampled_adata_list[i+1].obs_names[cell_idx]
                parent_names = downsampled_adata_list[i].obs_names[valid_parents].tolist()
                parents_dict[cell_name] = parent_names

        # Extend family information to full adata_list
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(downsampled_adata_list[i].obsm['spatial'])

        distances, indices = nn.kneighbors(adata_list[i].obsm['spatial'])
        
        full_children_dict = {}
        for idx, cell_name in enumerate(adata_list[i].obs_names):
            nearest_downsampled = downsampled_adata_list[i].obs_names[indices[idx][0]]
            if nearest_downsampled in children_dict:
                full_children_dict[cell_name] = children_dict[nearest_downsampled]

        adata_list[i].uns["children_dict"] = full_children_dict

        # Do the same for parents in the next section
        nn.fit(downsampled_adata_list[i+1].obsm['spatial'])
        distances, indices = nn.kneighbors(adata_list[i+1].obsm['spatial'])
        
        full_parents_dict = {}
        for idx, cell_name in enumerate(adata_list[i+1].obs_names):
            nearest_downsampled = downsampled_adata_list[i+1].obs_names[indices[idx][0]]
            if nearest_downsampled in parents_dict:
                full_parents_dict[cell_name] = parents_dict[nearest_downsampled]

        adata_list[i+1].uns["parents_dict"] = full_parents_dict

    return adata_list

def get_family_information(adata_list, section_ids, Pi_list, k=3):

    topk_mapping_AB_list = []
    topk_mapping_BA_list = []

    for Pi in Pi_list:
        topk_mapping_AB, topk_mapping_BA = get_topk_mapping(Pi, k=k)
        topk_mapping_AB_list.append(topk_mapping_AB)
        topk_mapping_BA_list.append(topk_mapping_BA)

    for i in range(len(section_ids)-1):
        adata_list[i].obsm["children"] = topk_mapping_AB_list[i]
        adata_list[i+1].obsm["parents"] = topk_mapping_BA_list[i].T

    for i in range(len(adata_list) - 1):
        # Create children dictionary for adata_list[i]
        children_dict = {}
        children = adata_list[i].obsm["children"]
        for cell_idx, child_indices in enumerate(children):
            valid_children = [idx for idx in child_indices if idx != -1]  # Assuming -1 represents no child
            if valid_children:
                cell_name = adata_list[i].obs_names[cell_idx]
                child_names = adata_list[i+1].obs_names[valid_children].tolist()
                children_dict[cell_name] = child_names
        
        # Store children dictionary in adata_list[i].uns
        adata_list[i].uns["children_dict"] = children_dict

        # Create parents dictionary for adata_list[i+1]
        parents_dict = {}
        parents = adata_list[i+1].obsm["parents"]
        for cell_idx, parent_indices in enumerate(parents):
            valid_parents = [idx for idx in parent_indices if idx != -1]  # Assuming -1 represents no parent
            if valid_parents:
                cell_name = adata_list[i+1].obs_names[cell_idx]
                parent_names = adata_list[i].obs_names[valid_parents].tolist()
                parents_dict[cell_name] = parent_names
        
        # Store parents dictionary in adata_list[i+1].uns
        adata_list[i+1].uns["parents_dict"] = parents_dict

    # Return the modified adata_list
    return adata_list

def create_ST2_adj_matrix(adata_list, section_ids):
    """
    Creates an adjacency matrix for ST2 data, incorporating spatial and temporal relationships.

    Parameters:
    -----------
    adata_list : list
        A list of AnnData objects, each representing data from a different section.
    section_ids : list
        Identifiers for each section corresponding to entries in adata_list.

    Returns:
    --------
    adata_concat : AnnData
        Concatenated AnnData object containing all sections.
    adj_concat : np.ndarray
        The final adjacency matrix incorporating both spatial and temporal relationships.
    """
    # Concatenate AnnData objects
    adata_concat = ad.concat(adata_list)
    print('adata_concat.shape: ', adata_concat.shape)

    adj_concat = adata_list[0].uns['adj']
    for batch_id in range(1, len(section_ids)):
        adj_concat = sp.block_diag((adj_concat, adata_list[batch_id].uns['adj']))
    batch_index = adata_concat.obs["batch_names"]

    # Update adjacency matrix with temporal relationships
    for i in range(len(section_ids) - 1):
        if "parents" in adata_list[i].obsm:
            prev_section_size = adata_list[i-1].shape[0] if i > 0 else 0
            current_section_size = adata_list[i].shape[0]
            adj_parents = np.zeros((prev_section_size, current_section_size))

            for j in range(current_section_size):
                parent_index = adata_list[i].obsm["parents"][j]
                if parent_index is not None:
                    adj_parents[parent_index, j] = 1

            mask_i_1 = batch_index == section_ids[i-1]
            mask_i = batch_index == section_ids[i]

            current_submatrix = adj_concat[np.ix_(mask_i_1, mask_i)]
            adj_concat[np.ix_(mask_i_1, mask_i)] = np.where(current_submatrix == 0, adj_parents, 1)

            current_submatrix = adj_concat[np.ix_(mask_i, mask_i_1)]
            adj_concat[np.ix_(mask_i, mask_i_1)] = np.where(current_submatrix == 0, adj_parents.T, 1)
        
        if "children" in adata_list[i].obsm:
            current_section_size = adata_list[i].shape[0]
            next_section_size = adata_list[i+1].shape[0] if i + 1 < len(adata_list) else 0
            adj_children = np.zeros((current_section_size, next_section_size))

            for j in range(current_section_size):
                child_index = adata_list[i].obsm["children"][j]
                if child_index is not None:
                    adj_children[j, child_index] = 1

            mask_i = batch_index == section_ids[i]
            mask_i_1 = batch_index == section_ids[i+1]

            adj_concat[np.ix_(mask_i, mask_i_1)] = adj_children
            adj_concat[np.ix_(mask_i_1, mask_i)] = adj_children.T

    # Store edge list in the concatenated AnnData object
    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

    return adata_concat, adj_concat

def STIFT(adata_list, section_ids, pre_n_epochs=500, n_epochs=2000, knn_cutoff=12, used_device='cuda:0'):

    Pi_list = []
    xi_list = []
    errs_list = []

    for i in range(len(adata_list)-1):
        Pi, xi, errs = align(adata_list[i], adata_list[i+1], alpha=0.2, gamma=50, epsilon=0.1, max_iter=200, 
                        balanced=False, use_gpu=True, normalize_xi=True, check_convergence=True, normalize_counts=True, normalize_spatial=True)
        Pi_list.append(Pi)
        xi_list.append(xi)
    torch.cuda.empty_cache()
    mapping_AB_list = []
    mapping_BA_list = []

    for Pi in Pi_list:
        mapping_AB, mapping_BA = get_mapping(Pi)
        mapping_AB_list.append(mapping_AB)
        mapping_BA_list.append(mapping_BA)

    for section_id in section_ids:
        print(section_id)
        adata = adata_list[section_ids.index(section_id)]

        # make spot name unique
        # adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

        STAligner.Cal_Spatial_Net(adata, model="KNN", k_cutoff=knn_cutoff)
        
        # Normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=10000) #ensure enough common HVGs in the combined matrix
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        adata_list[section_ids.index(section_id)] = adata[:, adata.var['highly_variable']]

    adata_list = get_family_information(adata_list, section_ids, Pi_list)

    adata_concat, adj_concat = create_ST2_adj_matrix(adata_list, section_ids)

    adata_concat = train_STIFT(adata_list=adata_list, adata_concat=adata_concat, hidden_dims=[512, 30], pre_n_epochs=pre_n_epochs, n_epochs=n_epochs, lr=0.001, key_added='STIFT', gradient_clipping=5., margin=1.0, weight_decay=0.0001, verbose=False,
                    random_seed=666, device=used_device)
    
    return adata_concat

def downsample_cells(adata, fraction=None, n_cells=None, random_state=1):
    """
    Downsample cells from an AnnData object based on a fraction or number of cells.

    Parameters:
    -----------
    adata : anndata.AnnData
        The original AnnData object.
    fraction : float, optional
        The fraction of cells to keep. Should be between 0 and 1.
    n_cells : int, optional
        The number of cells to keep. If greater than the total number of cells, all cells are kept.
    random_state : int, optional (default: 1)
        Random state for reproducibility.

    Returns:
    --------
    anndata.AnnData
        A new AnnData object with downsampled cells or the original if no downsampling is needed.
    """
    # Input validation
    if fraction is not None and n_cells is not None:
        raise ValueError("Provide either fraction or n_cells, not both.")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1.")
    if n_cells is not None and n_cells <= 0:
        raise ValueError("n_cells must be a positive integer.")
    
    # Calculate the number of cells to keep
    if fraction is not None:
        n_samples = int(adata.n_obs * fraction)
    elif n_cells is not None:
        n_samples = min(n_cells, adata.n_obs)  # Ensure n_samples doesn't exceed total cells
    else:
        raise ValueError("Either fraction or n_cells must be provided.")

    # If n_samples equals the total number of cells, return the original AnnData
    if n_samples == adata.n_obs:
        print("No downsampling needed. Returning the original AnnData object.")
        return adata

    # If fraction or n_cells results in 0 cells, keep at least 1 cell
    if n_samples == 0:
        print(f"Warning: Requested sampling results in 0 cells. Keeping 1 cell.")
        n_samples = 1

    # Use numpy to generate random indices
    np.random.seed(random_state)
    indices = np.random.choice(adata.n_obs, size=n_samples, replace=False)

    # Create a new AnnData object with the sampled cells
    adata_downsampled = adata[indices]
    
    return adata_downsampled

def get_family_information_from_downsample(downsampled_adata_list, adata_list, downsampled_Pi_list, section_ids, k=3):
    topk_mapping_AB_list = []
    topk_mapping_BA_list = []

    # Extract top-k mappings from the downsampled Pi matrices
    for Pi in downsampled_Pi_list:
        topk_mapping_AB, topk_mapping_BA = get_topk_mapping(Pi, k=k)
        topk_mapping_AB_list.append(topk_mapping_AB)
        topk_mapping_BA_list.append(topk_mapping_BA)

    for i in range(len(section_ids)-1):
        downsampled_adata_list[i].obsm["children"] = topk_mapping_AB_list[i]
        downsampled_adata_list[i+1].obsm["parents"] = topk_mapping_BA_list[i].T

    for i in range(len(adata_list) - 1):
        # Create children dictionary for downsampled adata_list[i]
        children_dict = {}
        children = downsampled_adata_list[i].obsm["children"]
        for cell_idx, child_indices in enumerate(children):
            valid_children = [idx for idx in child_indices if idx != -1]  # Assuming -1 represents no child
            if valid_children:
                cell_name = downsampled_adata_list[i].obs_names[cell_idx]
                child_names = downsampled_adata_list[i+1].obs_names[valid_children].tolist()
                children_dict[cell_name] = child_names

        # Create parents dictionary for downsampled adata_list[i+1]
        parents_dict = {}
        parents = downsampled_adata_list[i+1].obsm["parents"]
        for cell_idx, parent_indices in enumerate(parents):
            valid_parents = [idx for idx in parent_indices if idx != -1]  # Assuming -1 represents no parent
            if valid_parents:
                cell_name = downsampled_adata_list[i+1].obs_names[cell_idx]
                parent_names = downsampled_adata_list[i].obs_names[valid_parents].tolist()
                parents_dict[cell_name] = parent_names

        # Extend family information to full adata_list
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(downsampled_adata_list[i].obsm['spatial'])

        distances, indices = nn.kneighbors(adata_list[i].obsm['spatial'])
        
        full_children_dict = {}
        for idx, cell_name in enumerate(adata_list[i].obs_names):
            nearest_downsampled = downsampled_adata_list[i].obs_names[indices[idx][0]]
            if nearest_downsampled in children_dict:
                full_children_dict[cell_name] = children_dict[nearest_downsampled]

        adata_list[i].uns["children_dict"] = full_children_dict

        # Do the same for parents in the next section
        nn.fit(downsampled_adata_list[i+1].obsm['spatial'])
        distances, indices = nn.kneighbors(adata_list[i+1].obsm['spatial'])
        
        full_parents_dict = {}
        for idx, cell_name in enumerate(adata_list[i+1].obs_names):
            nearest_downsampled = downsampled_adata_list[i+1].obs_names[indices[idx][0]]
            if nearest_downsampled in parents_dict:
                full_parents_dict[cell_name] = parents_dict[nearest_downsampled]

        adata_list[i+1].uns["parents_dict"] = full_parents_dict

    return adata_list

def Cal_Spatial_Net_3d(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol', "imageheight"]

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G

