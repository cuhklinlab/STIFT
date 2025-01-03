import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import scanpy as sc

def interactive_3d_spatial_plot(adata_list, annotation_key="annotation", gene_of_interest=None, celltype_of_interest=None, title="3D Visualization", point_size=3, transparency=0.3, view_direction='default'):
    n_samples = len(adata_list)
    n_cols = n_samples
    n_rows = 1

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'scene'}] * n_cols] * n_rows,
        subplot_titles=[f"Sample {i + 1}" for i in range(n_samples)]
    )

    color_map = px.colors.qualitative.Plotly

    all_annotations = set()
    for adata in adata_list:
        all_annotations.update(adata.obs[annotation_key].unique())
    all_annotations = sorted(list(all_annotations))

    color_dict = {ann: color_map[i % len(color_map)] for i, ann in enumerate(all_annotations)}
    if celltype_of_interest:
        if isinstance(celltype_of_interest, str):
            celltype_of_interest = [celltype_of_interest]
        for ct in celltype_of_interest:
            color_dict[ct] = 'red'

    for i, adata in enumerate(adata_list):
        row = i // n_cols + 1
        col = i % n_cols + 1

        x = adata.obsm["spatial"][:, 0]
        y = adata.obsm["spatial"][:, 1]
        z = adata.obsm["spatial"][:, 2]

        annotations = adata.obs[annotation_key]

        if gene_of_interest and gene_of_interest in adata.var_names:
            color_values = adata[:, gene_of_interest].X
            if scipy.sparse.issparse(color_values):
                color_values = color_values.toarray()
            color_values = color_values.flatten()
            colorscale = 'Viridis'
            colorbar_title = f'{gene_of_interest} expression'

            scatter = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_values,
                    colorscale=colorscale,
                    opacity=1 - transparency,
                    colorbar=dict(title=colorbar_title)
                ),
                text=annotations,
                hoverinfo='text'
            )
            fig.add_trace(scatter, row=row, col=col)

        else:
            if celltype_of_interest:
                mask_of_interest = np.isin(annotations, celltype_of_interest)
                
                scatter = go.Scatter3d(
                    x=x[mask_of_interest], y=y[mask_of_interest], z=z[mask_of_interest],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color='red',
                        opacity=1 - transparency,
                    ),
                    name='Cells of Interest',
                    text=annotations[mask_of_interest],
                    hoverinfo='text',
                    showlegend=True
                )
                fig.add_trace(scatter, row=row, col=col)
                
                scatter = go.Scatter3d(
                    x=x[~mask_of_interest], y=y[~mask_of_interest], z=z[~mask_of_interest],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color='grey',
                        opacity=1 - transparency,
                    ),
                    name='Other Cells',
                    text=annotations[~mask_of_interest],
                    hoverinfo='text',
                    showlegend=True
                )
                fig.add_trace(scatter, row=row, col=col)
            else:
                for ann in all_annotations:
                    mask = annotations == ann
                    scatter = go.Scatter3d(
                        x=x[mask], y=y[mask], z=z[mask],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            color=color_dict[ann],
                            opacity=1 - transparency,
                        ),
                        name=ann,
                        text=ann,
                        hoverinfo='text',
                        showlegend=True
                    )
                    fig.add_trace(scatter, row=row, col=col)

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
        
        if view_direction == 'top':
            camera['eye'] = dict(x=0, y=0, z=2)
        elif view_direction == 'side':
            camera['eye'] = dict(x=2, y=0, z=0)
        elif view_direction == 'front':
            camera['eye'] = dict(x=0, y=2, z=0)

        fig.update_scenes(
            dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=camera
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=600 * n_rows,
        width=800 * n_cols,
        title_text=title,
        legend_title="Annotations",
    )

    return fig


# Example usage (for testing outside of this simplified example):
# fig = interactive_3d_spatial_plot(adata_list, gene_of_interest="GENE_NAME")
# fig = interactive_3d_spatial_plot(adata_list, celltype_of_interest="CELLTYPE_NAME")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_spatial_clusters(adata_concat, section_ids, cluster_key='louvain', save_dir=None):
    custom_colors = ['#FF7F00', '#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', 
                     '#7F7F7F', '#BCBD22', '#17BECF', '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', 
                     '#C5B0D5', '#C49C94', '#F7B6D2', '#C7C7C7', '#DBDB8D', '#9EDAE5', '#FFFF00', 
                     '#98FB98', '#FF69B4', '#00FFFF', '#FF1493', '#00FF00', '#FF00FF', '#1E90FF']
    
    celltype_list = adata_concat.obs[cluster_key].unique()
    sorted_celltype_list = sorted(celltype_list, key=lambda x: (x.isdigit(), x))

    fig, axs = plt.subplots(1, len(section_ids), figsize=(40, 6))
    
    for idx, batch in enumerate(section_ids):
        batch_data = adata_concat[adata_concat.obs['batch_names'] == batch]
        coordinatesA = batch_data.obsm['spatial'].copy()
        ax = axs[idx]
        
        for j, ann in enumerate(sorted_celltype_list):
            indices = batch_data.obs[cluster_key] == ann
            ax.scatter(coordinatesA[indices, 0], coordinatesA[indices, 1], s=10, 
                       c=[custom_colors[j % len(custom_colors)]], label=ann)
        
        ax.set_title(f" {section_ids[idx]}", fontsize=28, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()

    # Create custom legend handles with rectangles
    legend_handles = [mpatches.Rectangle((0, 0), 1, 1, fc=custom_colors[i % len(custom_colors)]) 
                      for i, _ in enumerate(sorted_celltype_list)]

    # Create legend
    fig.legend(legend_handles, sorted_celltype_list, loc='center left', bbox_to_anchor=(1, 0.5), 
               ncol=1, fontsize=12, frameon=False, title_fontsize=14)

    # Adjust figure size to accommodate legend
    fig.set_size_inches(44, 8)
    if save_dir != None:
        plt.savefig(save_dir, dpi=300, bbox_inches='tight')   
    plt.show()



def plot_spatial_specific_clusters(adata_concat, section_ids, cluster_key='louvain', cluster_to_plot=None):
    plt.figure(figsize=(25, 5))

    for idx, batch in enumerate(section_ids):
        # Create a square subplot
        plt.subplot(1, len(section_ids), idx+1)    
        batch_data = adata_concat[adata_concat.obs['batch_names'] == batch]
        coordinatesA = batch_data.obsm['spatial'].copy()

        # Plot all spots in grey
        plt.scatter(coordinatesA[:, 0], coordinatesA[:, 1], s=2, c='lightgrey', label='Other')

        if cluster_to_plot is not None:
            for ann in cluster_to_plot:
                indices = batch_data.obs[cluster_key] == ann
                plt.scatter(coordinatesA[indices, 0], coordinatesA[indices, 1], s=2, c='red', label=f'Cluster {ann}')
        
        plt.title(f"Section: {section_ids[idx]}")
        plt.axis('off')

    # Create a separate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.figlegend(by_label.values(), by_label.keys(), 
                  loc='center left', bbox_to_anchor=(1.05, 0.5), 
                  ncol=1)  # Adjust ncol as needed

    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    plt.show()

import random
import seaborn as sns

def plot_all_mappings_2D(adata_list, mapping_AB_list):
    n_plots = len(adata_list) - 1
    fig, axs = plt.subplots(2, n_plots // 2 + 1, figsize=(15 * (n_plots // 2 + 1), 20))
    
    color_palette = sns.color_palette("colorblind", 30)
    all_annotations = np.unique(np.concatenate([adata.obs["annotation"] for adata in adata_list]))
    color_dict = {ann: color_palette[i % len(color_palette)] for i, ann in enumerate(all_annotations)}

    for i in range(n_plots):
        adata1 = adata_list[i].copy()
        adata2 = adata_list[i+1].copy()
        mapping_AB = mapping_AB_list[i]

        # Adjust x-coordinate of adata2
        adata2.obsm['spatial'][:, 0] += 400

        row = i // (n_plots // 2 + 1)
        col = i % (n_plots // 2 + 1)
        ax = axs[row, col] if n_plots > 1 else axs

        for ann in np.unique(adata1.obs["annotation"]):
            mask = adata1.obs["annotation"] == ann
            ax.scatter(adata1.obsm['spatial'][mask, 0], 
                       adata1.obsm['spatial'][mask, 1],
                       c=[color_dict[ann]], alpha=0.3, s=1, label=f'Dataset {i+1} - {ann}')

        for ann in np.unique(adata2.obs["annotation"]):
            mask = adata2.obs["annotation"] == ann
            ax.scatter(adata2.obsm['spatial'][mask, 0], 
                       adata2.obsm['spatial'][mask, 1],
                       c=[color_dict[ann]], alpha=0.3, s=1, label=f'Dataset {i+2} - {ann}')

        k = 50
        selected_indices = random.sample(range(len(adata1.obs_names)), k)

        for idx in selected_indices:
            ax.plot([adata1.obsm['spatial'][idx, 0], adata2.obsm['spatial'][mapping_AB[idx], 0]],
                    [adata1.obsm['spatial'][idx, 1], adata2.obsm['spatial'][mapping_AB[idx], 1]],
                    color="black", linestyle='-', linewidth=1, alpha=1)

        ax.set_title(f"Alignment {i+1} to {i+2}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import seaborn as sns

def plot_all_mappings(adata_list, mapping_AB_list, section_ids):
    n_plots = len(adata_list) - 1
    fig = plt.figure(figsize=(5 * (n_plots), 10))
    
    color_palette = sns.color_palette("colorblind", 30)
    all_annotations = np.unique(np.concatenate([adata.obs["annotation"] for adata in adata_list]))
    color_dict = {ann: color_palette[i % len(color_palette)] for i, ann in enumerate(all_annotations)}

    for i in range(n_plots):
        adata1 = adata_list[i].copy()
        adata2 = adata_list[i+1].copy()
        mapping_AB = mapping_AB_list[i]

        adata1.obsm["spatial"] = np.concatenate([adata1.obsm["spatial"], np.zeros((adata1.obsm["spatial"].shape[0], 1))], axis=1)
        adata2.obsm["spatial"] = np.concatenate([adata2.obsm["spatial"], np.ones((adata2.obsm["spatial"].shape[0], 1))*10], axis=1)

        ax = fig.add_subplot(1, n_plots, i+1, projection='3d')

        for ann in np.unique(adata1.obs["annotation"]):
            mask = adata1.obs["annotation"] == ann
            ax.scatter(adata1.obsm['spatial'][mask, 0], 
                       adata1.obsm['spatial'][mask, 1],
                       adata1.obsm['spatial'][mask, 2], 
                       c=[color_dict[ann]], alpha=0.3, s=1, label=f'Dataset {i+1} - {ann}')

        for ann in np.unique(adata2.obs["annotation"]):
            mask = adata2.obs["annotation"] == ann
            ax.scatter(adata2.obsm['spatial'][mask, 0], 
                       adata2.obsm['spatial'][mask, 1],
                       adata2.obsm['spatial'][mask, 2], 
                       c=[color_dict[ann]], alpha=0.1, s=1, label=f'Dataset {i+2} - {ann}')

        k = 500
        selected_indices = random.sample(range(len(adata1.obs_names)), k)

        for idx in selected_indices:
            ax.plot([adata1.obsm['spatial'][idx, 0], adata2.obsm['spatial'][mapping_AB[idx], 0]],
                    [adata1.obsm['spatial'][idx, 1], adata2.obsm['spatial'][mapping_AB[idx], 1]],
                    [adata1.obsm['spatial'][idx, 2], adata2.obsm['spatial'][mapping_AB[idx], 2]],
                    color="black", linestyle='-', linewidth=1, alpha=0.05)

        ax.set_title(f"Alignment {section_ids[i]} to {section_ids[i+1]}")
        
        # Remove axes, ticks, and labels
        ax.set_axis_off()
        
        ax.view_init(elev=10, azim=-90)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

def plot_cluster_and_markers(adata_concat, cluster_of_interest, markers_of_interest, cluster_key="leiden_annotation"):
    # Ensure leiden_annotation is string type
    adata_concat.obs[cluster_key] = adata_concat.obs[cluster_key].astype(str)
    adata_concat.obs['annotation'] = adata_concat.obs['annotation'].astype(str)
    stages = adata_concat.obs['batch_names'].unique()

    # Create a combined set of all unique values from leiden_annotation, annotation, and batch_names
    all_categories = set(adata_concat.obs[cluster_key].unique()) | set(adata_concat.obs['annotation'].unique()) | set(adata_concat.obs['batch_names'].unique())

    # Create a color mapping for all categories
    n_categories = len(all_categories)
    color_palette = sns.color_palette("husl", n_colors=n_categories)
    color_dict = dict(zip(all_categories, color_palette))

    # Set up the plot parameters
    fig_size = 20

    for cell_type, marker in zip(cluster_of_interest, markers_of_interest):
        fig, axes = plt.subplots(2, 5, figsize=(fig_size, fig_size/2))
        fig.suptitle(f'{cell_type} - {marker}', fontsize=20)
        
        for idx, stage in enumerate(stages):
            stage_data = adata_concat[adata_concat.obs['batch_names'] == stage]
            
            # Upper subplot: Cell type distribution (leiden_annotation)
            ax_upper = axes[0, idx]
            sc.pl.spatial(stage_data, 
                          color=cluster_key,
                          groups=[cell_type],
                          ax=ax_upper, 
                          show=False, 
                          title=f'Leiden in {stage}',
                          size=1,
                          spot_size=2,
                          na_color='lightgrey',
                          palette=color_dict)
            ax_upper.set_xticks([])
            ax_upper.set_yticks([])

            # Lower subplot: Gene expression
            ax_lower = axes[1, idx]
            sc.pl.spatial(stage_data, 
                          color=marker, 
                          ax=ax_lower, 
                          show=False, 
                          title=f'{marker} Expression in {stage}',
                          color_map='viridis',
                          size=1,
                          spot_size=2)
            ax_lower.set_xticks([])
            ax_lower.set_yticks([])
        
        plt.tight_layout()
        plt.show()

