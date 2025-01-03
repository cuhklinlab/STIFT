import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from .STALIGNER import STAligner

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset


def train_STIFT(adata_list=None, adata_concat=None, hidden_dims=[512, 30], pre_n_epochs=500, n_epochs=1000, lr=0.001, key_added='STIFT', weight_triplet=1, gradient_clipping=5., margin=1.0, weight_decay=0.0001, verbose=False,
                    random_seed=666, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata_concat.obs['batch_names'].unique())
    edgeList = adata_concat.uns['edgeList']
    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                prune_edge_index=torch.LongTensor(np.array([])),
                x=torch.FloatTensor(adata_concat.X.todense()))
    data = data.to(device)

    model = STAligner(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)
    losses = []
    mse_losses = []
    triplet_losses = []
    print('Pretrain with STAGATE...')
    for epoch in tqdm(range(pre_n_epochs)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)

        loss = F.mse_loss(data.x, out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    with torch.no_grad():
        z, out = model(data.x, data.edge_index)
    STAGATE_rep = z.to('cpu').detach().numpy()
    adata_concat.obsm[key_added] = STAGATE_rep

    print('Train with STIFT...')
    for epoch in tqdm(range(n_epochs)):
        if epoch % 100 == 0:
            if verbose:
                print('Update spot triplets at epoch ' + str(epoch))
            adata_concat.obsm['STAGATE'] = z.cpu().detach().numpy()

            # If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
            # not all points have MNN anchors
            family_dict = create_family_dicts(adata_list, section_ids)
            anchor_ind, positive_ind, negative_ind = create_triplets(adata_concat, family_dict, section_ids)

        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        mse_loss = F.mse_loss(data.x, out)

        anchor_arr = z[anchor_ind,]
        positive_arr = z[positive_ind,]
        negative_arr = z[negative_ind,]

        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

        loss = mse_loss + weight_triplet * tri_output
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        if epoch % 100 == 0:
            losses.append(loss.item())
            mse_losses.append(mse_loss.item())
            triplet_losses.append(weight_triplet * tri_output.item())
            print('loss: ', loss.item(), 'mse_loss: ', mse_loss.item(), 'triplet_loss: ', weight_triplet * tri_output.item()) 
    model.eval()
    adata_concat.obsm[key_added] = z.cpu().detach().numpy()
    return adata_concat

def create_family_dicts(adata_list, section_ids):
    family_dict = {}

    for i in range(len(adata_list) - 1):
        batch_pair = f"{section_ids[i]}_{section_ids[i+1]}"
        family_dict[batch_pair] = {}

        # Process children
        if 'children_dict' in adata_list[i].uns:
            children_dict = adata_list[i].uns['children_dict']
            for cell_name, child_names in children_dict.items():
                family_dict[batch_pair][cell_name] = child_names

        # Process parents
        if 'parents_dict' in adata_list[i+1].uns:
            parents_dict = adata_list[i+1].uns['parents_dict']
            for cell_name, parent_names in parents_dict.items():
                family_dict[batch_pair][cell_name] = parent_names

    return family_dict

def create_triplets(adata_concat, family_dict, section_ids):
    anchor_ind = []
    positive_ind = []
    negative_ind = []

    for batch_pair in family_dict.keys():
        batchname_list = adata_concat.obs['batch_names'][family_dict[batch_pair].keys()]

        cellname_by_batch_dict = dict()
        for batch_id in range(len(section_ids)):
            cellname_by_batch_dict[section_ids[batch_id]] = adata_concat.obs_names[
                adata_concat.obs['batch_names'] == section_ids[batch_id]].values

        anchor_list = []
        positive_list = []
        negative_list = []

        for anchor in family_dict[batch_pair].keys():
            anchor_list.append(anchor)
            
            # Select the first child/parent as positive
            positive_spot = family_dict[batch_pair][anchor][0]
            positive_list.append(positive_spot)

            # Select a random cell from the same batch as the anchor for negative
            anchor_batch = adata_concat.obs.loc[anchor, 'batch_names']
            section_size = len(cellname_by_batch_dict[anchor_batch])
            negative_list.append(
                cellname_by_batch_dict[anchor_batch][np.random.randint(section_size)]
            )
            
        # Convert cell names to indices
        batch_as_dict = dict(zip(list(adata_concat.obs_names), range(0, adata_concat.shape[0])))
        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

    return anchor_ind.astype(int), positive_ind.astype(int), negative_ind.astype(int)