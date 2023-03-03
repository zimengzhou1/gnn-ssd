import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import os.path as osp
import os
import argparse
import scipy
import numpy as np
import json
import time

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from customNeighborSampler import MMAPNeighborSampler

from lib.utils import *

__file__ = os.path.abspath('')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set environment and path
os.environ['GINEX_NUM_THREADS'] = str(128)

dataset_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Reddit')
dataset = Reddit(dataset_path)

# # Construct sparse formats
# print('Creating coo/csc/csr format of dataset...')
# num_nodes = dataset[0].num_nodes
# coo = dataset[0].edge_index.numpy()
# v = np.ones_like(coo[0])
# coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
# csc = coo.tocsc()
# csr = coo.tocsr()
# print('Done!')


# # Save csc-formatted dataset
# indptr = csc.indptr.astype(np.int64)
# indices = csc.indices.astype(np.int64)
# features = dataset[0].x
# labels = dataset[0].y

# os.makedirs(dataset_path, exist_ok=True)
# indptr_path = os.path.join(dataset_path, 'indptr.dat')
# indices_path = os.path.join(dataset_path, 'indices.dat')
# features_path = os.path.join(dataset_path, 'features.dat')
# labels_path = os.path.join(dataset_path, 'labels.dat')
# conf_path = os.path.join(dataset_path, 'conf.json')
# split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

# print('Saving indptr...')
# indptr_mmap = np.memmap(indptr_path, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
# indptr_mmap[:] = indptr[:]
# indptr_mmap.flush()
# print('Done!')

# print('Saving indices...')
# indices_mmap = np.memmap(indices_path, mode='w+', shape=indices.shape, dtype=indices.dtype)
# indices_mmap[:] = indices[:]
# indices_mmap.flush()
# print('Done!')

# print('Saving features...')
# features_mmap = np.memmap(features_path, mode='w+', shape=dataset[0].x.shape, dtype=np.float32)
# features_mmap[:] = features[:]
# features_mmap.flush()
# print('Done!')

# print('Saving labels...')
# labels = labels.type(torch.float32)
# labels_mmap = np.memmap(labels_path, mode='w+', shape=dataset[0].y.shape, dtype=np.float32)
# labels_mmap[:] = labels[:]
# labels_mmap.flush()
# print('Done!')

# print('Making conf file...')
# mmap_config = dict()
# mmap_config['num_nodes'] = int(dataset[0].num_nodes)
# mmap_config['indptr_shape'] = tuple(indptr.shape)
# mmap_config['indptr_dtype'] = str(indptr.dtype)
# mmap_config['indices_shape'] = tuple(indices.shape)
# mmap_config['indices_dtype'] = str(indices.dtype)
# mmap_config['indices_shape'] = tuple(indices.shape)
# mmap_config['indices_dtype'] = str(indices.dtype)
# mmap_config['indices_shape'] = tuple(indices.shape)
# mmap_config['indices_dtype'] = str(indices.dtype)
# mmap_config['features_shape'] = tuple(features_mmap.shape)
# mmap_config['features_dtype'] = str(features_mmap.dtype)
# mmap_config['labels_shape'] = tuple(labels_mmap.shape)
# mmap_config['labels_dtype'] = str(labels_mmap.dtype)
# mmap_config['num_classes'] = int(dataset.num_classes)
# json.dump(mmap_config, open(conf_path, 'w'))
# print('Done!')

# print('Saving split index...')
# splits = {'train': dataset[0].train_mask, 'test': dataset[0].test_mask, 'valid': dataset[0].val_mask}
# torch.save(splits, split_idx_path)
# print('Done!')

# # Calculate and save score for neighbor cache construction
# print('Calculating score for neighbor cache construction...')
# score_path = os.path.join(dataset_path, 'nc_score.pth')
# csc_indptr_tensor = torch.from_numpy(csc.indptr.astype(np.int64))
# csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))

# eps = 0.00000001
# in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1]) + eps
# out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps
# score = out_num_neighbors / in_num_neighbors
# print('Done!')

# print('Saving score...')
# torch.save(score, score_path)
# print('Done!')

def get_mmap_dataset(path='../data/Reddit'):
    indptr_path = os.path.join(path, 'indptr.dat')
    indices_path = os.path.join(path, 'indices.dat')
    features_path = os.path.join(path, 'features.dat')
    labels_path = os.path.join(path, 'labels.dat')
    conf_path = os.path.join(path, 'conf.json')
    split_idx_path = os.path.join(path, 'split_idx.pth')

    conf = json.load(open(conf_path, 'r'))

    # Assume we only memmap for large files - the adjacency matrix (indices) + features ~ 13GB and 50GB respectively

    indptr = np.fromfile(indptr_path, dtype=conf['indptr_dtype']).reshape(tuple(conf['indptr_shape']))
    indices = np.memmap(indices_path, mode='r', shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype'])
    features_shape = conf['features_shape']
    features = np.memmap(features_path, mode='r', shape=tuple(features_shape), dtype=conf['features_dtype'])
    labels = np.fromfile(labels_path, dtype=conf['labels_dtype'], count=conf['num_nodes']).reshape(tuple([conf['labels_shape'][0]]))

    indptr = torch.from_numpy(indptr)
    indices = torch.from_numpy(indices)
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)

    num_nodes = conf['num_nodes']
    num_features = conf['features_shape'][1]
    num_classes = conf['num_classes']

    split_idx = torch.load(split_idx_path)
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    return indptr, indices, features, labels, num_features, num_classes, num_nodes, train_idx, val_idx, test_idx


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--num-workers', type=int, default=32)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--sizes', type=str, default='10,10')
argparser.add_argument('--ginex-num-threads', type=int, default=128)
argparser.add_argument('--train-only', dest='train_only', default=True, action='store_true')
args = argparser.parse_args()

indptr, indices, x, y, num_features, num_classes, num_nodes, train_idx, valid_idx, test_idx = get_mmap_dataset()

sizes = [int(size) for size in args.sizes.split(',')]
train_loader = MMAPNeighborSampler(indptr, indices, node_idx=train_idx,
                               sizes=sizes, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
test_loader = MMAPNeighborSampler(indptr, indices, node_idx=test_idx,
                               sizes=sizes, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)
valid_loader = MMAPNeighborSampler(indptr, indices, node_idx=valid_idx,
                               sizes=sizes, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

# Define model
device = torch.device('cuda:%d' % args.gpu)
torch.cuda.set_device(device)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.num_layers = 2

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def reset_parameters(self):
      for conv in self.convs:
        conv.reset_parameters()

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

model = SAGE(num_features, args.num_hiddens, num_classes)
model = model.to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(sum(train_idx)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    # Sample
    for step, (batch_size, ids, adjs) in enumerate(train_loader):
        # Gather
        batch_inputs = gather_mmap(x, ids)
        batch_labels = y[ids[:batch_size]]

        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)

        adjs_cuda = [adj.to(device) for adj in adjs]

        # Forward
        out = model(batch_inputs_cuda, adjs_cuda)
        loss = F.nll_loss(out, batch_labels_cuda.long())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Free
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        tensor_free(batch_inputs)
        del(batch_inputs_cuda)
        del(ids)
        del(adjs)
        del(batch_labels_cuda)
        torch.cuda.empty_cache()
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def inference(mode='test'):
    model.eval()

    if mode == 'test':
        pbar = tqdm(total=test_idx.size(0))
    elif mode == 'valid':
        pbar = tqdm(total=valid_idx.size(0))
    pbar.set_description('Evaluating')

    total_loss = total_correct = 0

    if mode == 'test':
        inference_loader = test_loader
    elif mode == 'valid':
        inference_loader = valid_loader

    # Sample
    for step, (batch_size, ids, adjs) in enumerate(inference_loader):
        # Gather
        batch_inputs = gather_mmap(x, ids)
        batch_labels = y[ids[:batch_size]]

        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs = [adj.to(device) for adj in adjs]

        # Forward
        out = model(batch_inputs_cuda, adjs)
        loss = F.nll_loss(out, batch_labels_cuda.long())
        tensor_free(batch_inputs)

        torch.cuda.synchronize()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(inference_loader)
    if mode == 'test':
        approx_acc = total_correct / test_idx.size(0)
    elif mode == 'valid':
        approx_acc = total_correct / valid_idx.size(0)

    return loss, approx_acc


model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
best_val_acc = final_test_acc = 0
for epoch in range(args.num_epochs):
    start = time.time()
    loss, acc = train(epoch)
    end = time.time()
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    print('Epoch time: {:.4f} ms'.format((end - start) * 1000))

    if epoch > 3 and not args.train_only:
        val_loss, val_acc = inference(mode='valid')
        test_loss, test_acc = inference(mode='test')
        print ('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(val_loss, val_acc, test_loss, test_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc

if not args.train_only:
    print('Final Test acc: {final_test_acc:.4f}')