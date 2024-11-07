import argparse
import time
import os, sys
import os.path as osp
from shutil import copy
import copy as cp
from tqdm import tqdm
import pdb
import utils
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import coarsening
from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected
from logger import LightLogging
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from models import SEAL, GCN, GraphSAGE, NGNN
import logging

if logging.getLogger(__name__):
    logger = logging.getLogger(__name__)
else: 
    logger = LightLogging('../logger', log_name='cgrl', log_level='info')
Logger = logger.info
def train():
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test():
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]
    
    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'rocauc':
        results = evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results

def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)
    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results

def evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
        })[f'rocauc']

    test_rocauc = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'rocauc']

    results = {}
    results['rocauc'] = (valid_rocauc, test_rocauc)
    return results

# Data settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbl-collab')

# GNN settings
parser.add_argument('--model', type=str, default='DGCNN')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
# Subgraph extraction settings
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true', 
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
# Training settings
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)

parser.add_argument('--num_workers', type=int, default=16, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true', 
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true', 
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--only_test', action='store_true', 
                    help="only test without training")
parser.add_argument('--test_multiple_models', action='store_true', 
                    help="test multiple models together")
parser.add_argument('--coarsening_ratio', type=float, default=0.9)
parser.add_argument('--coarsening_method', type=str, default='lam')
args = parser.parse_args()

if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'

args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not args.keep_old:
    # Backup python files.
    copy('seal_link_pred.py', args.res_dir)
    copy('utils.py', args.res_dir)
log_file = os.path.join(args.res_dir, 'log.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)

if args.dataset.startswith('ogbl'):
    dataset = PygLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    if args.dataset.startswith('ogbl-vessel'):
        # normalize node features
        data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
        data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
        data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)
else:
    path = osp.join('dataset', args.dataset)
    dataset = Planetoid(path, args.dataset)
    split_edge = utils.do_edge_split(dataset, args.fast_split)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()
data = coarsening.coarsen(data, 1-args.coarsening_ratio, args.coarsening_method)
if args.dataset.startswith('ogbl-citation'):
    args.eval_metric = 'mrr'
    directed = True
elif args.dataset.startswith('ogbl-vessel'):
    args.eval_metric = 'rocauc'
    directed = False
elif args.dataset.startswith('ogbl'):
    args.eval_metric = 'hits'
    directed = False
else:  # assume other datasets are undirected
    args.eval_metric = 'auc'
    directed = False

if args.use_valedges_as_input:
    val_edge_index = split_edge['valid']['edge'].t()
    if not directed:
        val_edge_index = to_undirected(val_edge_index)
    data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    if 'edge_weight' in data:
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
        data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

if args.dataset.startswith('ogbl'):
    evaluator = Evaluator(name=args.dataset)
if args.eval_metric == 'hits':
    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
elif args.eval_metric == 'mrr':
    loggers = {
        'MRR': Logger(args.runs, args),
    }
elif args.eval_metric == 'rocauc':
    loggers = {
        'rocauc': Logger(args.runs, args),
    }

elif args.eval_metric == 'auc':
    loggers = {
        'AUC': Logger(args.runs, args),
    }
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.use_heuristic:
    # Test link prediction heuristics.
    num_nodes = data.num_nodes
    if 'edge_weight' in data:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
                       shape=(num_nodes, num_nodes))

    pos_val_edge, neg_val_edge = utils.get_pos_neg_edges('valid', split_edge, 
                                                   data.edge_index, 
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = utils.get_pos_neg_edges('test', split_edge, 
                                                     data.edge_index, 
                                                     data.num_nodes)
    pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
    neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
    pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
    neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'rocauc':
        results = evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        val_pred = torch.cat([pos_val_pred, neg_val_pred])
        val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                              torch.zeros(neg_val_pred.size(0), dtype=int)])
        test_pred = torch.cat([pos_test_pred, neg_test_pred])
        test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                              torch.zeros(neg_test_pred.size(0), dtype=int)])
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    for key, result in results.items():
        loggers[key].add_result(0, result)
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    pdb.set_trace()
    exit()


path = dataset.root + '_cgrl{}'.format(args.data_appendix)
use_coalesce = True if args.dataset == 'ogbl-collab' else False
if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
    args.num_workers = 0

dataset_class = 'cgrlD'
train_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.train_percent, 
    split='train', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
) 

val_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.val_percent, 
    split='valid', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
)
test_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.test_percent, 
    split='test', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    directed=directed, 
)

max_z = 1000  # set a large max_z so that every z has embeddings to look up

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                        num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                         num_workers=args.num_workers)

if args.train_node_embedding:
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
elif args.pretrained_node_embedding:
    weight = torch.load(args.pretrained_node_embedding)
    emb = torch.nn.Embedding.from_pretrained(weight)
    emb.weight.requires_grad=False
else:
    emb = None

for run in range(args.runs):
    if args.model == 'SEAL':
        model = SEAL(args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
                      train_dataset, args.dynamic_train, use_feature=args.use_feature, 
                      node_embedding=emb).to(device)
    elif args.model == 'SAGE':
        model = GraphSAGE(args.hidden_channels, args.num_layers, max_z, train_dataset,  
                     args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'GCN':
        model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    elif args.model == 'NGNN':
        model = NGNN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb).to(device)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')
    with open(log_file, 'a') as f:
        print(f'Total number of parameters is {total_params}', file=f)
        if args.model == 'DGCNN':
            print(f'SortPooling k is set to {model.k}', file=f)

    start_epoch = 1
    if args.continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_model_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_optimizer_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        start_epoch = args.continue_from + 1
        args.epochs -= args.continue_from
    
    if args.only_test:
        results = test()
        for key, result in results.items():
            loggers[key].add_result(run, result)
        for key, result in results.items():
            valid_res, test_res = result
            print(key)
            print(f'Run: {run + 1:02d}, '
                  f'Valid: {100 * valid_res:.2f}%, '
                  f'Test: {100 * test_res:.2f}%')
        pdb.set_trace()
        exit()

    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss = train()

        if epoch % args.eval_steps == 0:
            results = test()
            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                model_name = os.path.join(
                    args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, epoch))
                optimizer_name = os.path.join(
                    args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run+1, epoch))
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

                for key, result in results.items():
                    valid_res, test_res = result
                    to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                                f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                f'Test: {100 * test_res:.2f}%')
                    print(key)
                    print(to_print)
                    with open(log_file, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(run)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(run, f=f)

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics()
    with open(log_file, 'a') as f:
        print(key, file=f)
        loggers[key].print_statistics(f=f)
print(f'Total number of parameters is {total_params}')
print(f'Results are saved in {args.res_dir}')

