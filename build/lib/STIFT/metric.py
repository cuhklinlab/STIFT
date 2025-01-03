import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import silhouette_samples, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

def batch_entropy_mixing_score(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Calculate batch entropy mixing score
    
    Algorithm
    ---------
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.
    
    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.
        
    Returns
    -------
    Batch entropy mixing score
    """
#     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i]/P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i]/P[i])/a
            entropy = entropy - adapt_p[i]*np.log(adapt_p[i]+10**-8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
            P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
                                                 [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))

def compute_integration_metrics(adata, embedding_key, batch_key, cluster_key, cell_type_key, spatial_key="spatial"):
    """
    Compute integration metrics for the given AnnData object.
    
    Parameters:
    - adata: AnnData object
    - embedding_key: Key for the integrated embedding in adata.obsm
    - batch_key: Key for batch information in adata.obs
    - cell_type_key: Key for cell type information in adata.obs
    - spatial_key: Key for spatial coordinates in adata.obsm (optional, for SCS)
    
    Returns:
    - dict: A dictionary containing the computed metrics
    """
    embedding = adata.obsm[embedding_key]
    batches = adata.obs[batch_key].astype('category')
    cell_types = adata.obs[cell_type_key].astype('category')
    cluster_labels = adata.obs[cluster_key].astype('category')

    n_batches = len(batches.cat.categories)
    n_spots = len(embedding)

    # 1. Batch Entropy Score
    batch_entropy = batch_entropy_mixing_score(
        data=embedding,
        batches=batches.cat.codes,
        n_neighbors=100,
        n_pools=100,
        n_samples_per_pool=100
    )
    
    # 2. Batch Average Silhouette Width (ASW)
    batch_silhouette = silhouette_samples(embedding, batches.cat.codes)
    batch_asw = np.mean(1 - np.abs(batch_silhouette))
    
    # 3. Silhouette (Cell-type ASW)
    cell_type_silhouette = silhouette_samples(embedding, cell_types.cat.codes)
    silhouette = 0.5 * (np.mean(cell_type_silhouette) + 1)

    # 4. Local Inverse Simpson Index (iLISI and cLISI)
    def compute_lisi(X, labels, perplexity=30):
        nn = NearestNeighbors(n_neighbors=min(100, n_spots-1), metric='euclidean')
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        lisi_scores = []
        for i in range(n_spots):
            d = distances[i]
            ind = indices[i]
            
            p = np.exp(-d / (2 * np.square(d[perplexity])))
            p /= np.sum(p)
            
            label_counts = np.bincount(labels[ind], weights=p)
            lisi = 1 / np.sum(np.square(label_counts[label_counts > 0]))
            lisi_scores.append(lisi)
        
        return np.mean(lisi_scores)
    
    ilisi = compute_lisi(embedding, batches.cat.codes)
    clisi = compute_lisi(embedding, cell_types.cat.codes)

    # 5. Adjusted Rand Index (ARI)
    # Note: ARI is calculated between cluster labels and cell types
    ari = adjusted_rand_score(cluster_labels, cell_types)

    return {
        'Batch Entropy Score': batch_entropy,
        'Batch ASW': batch_asw,
        'Silhouette (Cell-type ASW)': silhouette,
        'iLISI': ilisi,
        'cLISI': clisi,
        'ARI': ari
    }

def get_evaluation_dataframe(adata_concat, section_ids, embedding_key, batch_key, cluster_key, cell_type_key):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import silhouette_samples, adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors

    # Initialize an empty DataFrame to store all metrics
    all_metrics_df = pd.DataFrame()

    # Compute metrics for the entire dataset
    metrics_all = compute_integration_metrics(adata_concat, embedding_key, cluster_key, batch_key, cell_type_key)
    metrics_all_df = pd.DataFrame([metrics_all])  # Convert dict to DataFrame
    metrics_all_df['Section'] = 'All'
    all_metrics_df = pd.concat([all_metrics_df, metrics_all_df], ignore_index=True)

    # Compute metrics for each section
    for section_id in section_ids:
        adata_section = adata_concat[adata_concat.obs['batch_names'] == section_id]
        metrics = compute_integration_metrics(adata_section, embedding_key, cluster_key, batch_key, cell_type_key)
        metrics_df = pd.DataFrame([metrics])  # Convert dict to DataFrame
        metrics_df['Section'] = section_id
        all_metrics_df = pd.concat([all_metrics_df, metrics_df], ignore_index=True)

    # Display the resulting dataframe
    print(all_metrics_df)
    return all_metrics_df


def calculate_cluster_annotations(adata, cluster_key='leiden', annotation_key='annotation'):
    # Create a cross-tabulation of clusters and annotations
    cross_tab = pd.crosstab(adata.obs[cluster_key], adata.obs[annotation_key])
    
    # Find the most common annotation for each cluster
    cluster_to_annotation = {}
    for cluster in cross_tab.index:
        max_count = cross_tab.loc[cluster].max()
        max_annotations = cross_tab.loc[cluster][cross_tab.loc[cluster] == max_count].index.tolist()
        if len(max_annotations) == 1:
            cluster_to_annotation[cluster] = max_annotations[0]
        else:
            # If there's a tie, assign the cluster number as the annotation
            cluster_to_annotation[cluster] = f"Cluster_{cluster}"
    
    # Create a new column in obs with the assigned annotation for each cluster
    new_annotation_key = f'{cluster_key}_annotation'
    adata.obs[new_annotation_key] = adata.obs[cluster_key].astype(str).map(cluster_to_annotation)
    
    # Ensure the new column is a string type, not categorical
    adata.obs[new_annotation_key] = adata.obs[new_annotation_key].astype(str)
    
    # Print the mapping
    print(f"{cluster_key.capitalize()} cluster to annotation mapping:")
    for cluster, annotation in cluster_to_annotation.items():
        print(f"Cluster {cluster}: {annotation}")
    
    # Calculate the accuracy of this assignment
    accuracy = np.mean(adata.obs[annotation_key].astype(str) == adata.obs[new_annotation_key])
    print(f"\nAccuracy of {cluster_key.capitalize()} cluster annotation: {accuracy:.2%}")
    
    # Print the cross-tabulation table
    print(f"\nCross-tabulation of {cluster_key.capitalize()} clusters and annotations:")
    print(cross_tab)
    
    return cross_tab, accuracy, cluster_to_annotation

# Usage example:
# cross_tab, accuracy = calculate_cluster_annotations(adata_concat, cluster_key='leiden', annotation_key='annotation')