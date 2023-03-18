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
from cache import FeatureCache

from lib.utils import *

__file__ = os.path.abspath('')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set environment and path
os.environ['GINEX_NUM_THREADS'] = str(128)

dataset_path = "/mnt/raid0nvme1/zz/ginex/data/Reddit"

def get_mmap_dataset(path=dataset_path):
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

    return indptr, indices, features, labels, num_features, num_classes, num_nodes, train_idx, val_idx, test_idx, features_path


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
argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
args = argparser.parse_args()

indptr, indices, x, y, num_features, num_classes, num_nodes, train_idx, valid_idx, test_idx, features_path = get_mmap_dataset()
#x = x.cpu()
feature_cache_size = int(len(x)/(100/30))
mmapped_features = x
print("num features: ", num_features)
print(len(x[0]))

score_path = os.path.join(dataset_path, 'out_deg.pth')
score = torch.load(score_path)
sorted_indices = score
print("sorted len: ", len(sorted_indices))

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

# Initialized cache
cache = FeatureCache(feature_cache_size, num_nodes, mmapped_features, num_features, False)

# Fill cache with top out-degree
indices = []
for i in range(feature_cache_size):
    indices.append(sorted_indices[i])

indices = torch.tensor(indices, dtype=torch.long).cpu()
cache.fill_cache(indices)

print("cache filled!")

def train(epoch):
    model.train()

    pbar = tqdm(total=int(sum(train_idx)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    # Sample
    for step, (batch_size, ids, adjs) in enumerate(train_loader):
        # Gather
        #batch_inputs = gather_mmap(x, ids)
        batch_inputs, hits, misses = gather_ginex(features_path, ids, num_features, cache)
        #print("hits: ", hits, " misses: ", misses)
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