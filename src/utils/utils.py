# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import numpy as np
import torch
import importlib
import datetime
import random
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter, segment
from typing import Optional
from torch import Tensor
import torch
import torch_geometric
import networkx as nx
from scipy.spatial import Delaunay



def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')

def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    return dgl_graph

def init_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
        
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False 
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS WORKSPACE CONFIG']=':16:8'


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

import umap
import torch
import umap
from scipy.spatial import Delaunay

def build_sparse_delaunay_graph(feat,norm_type='sym' ):
    """
    根据 feat 构造 Delaunay 图，并通过 get_sparse_laplacian 对边权进行处理，
    返回一个稀疏的 Laplacian 邻接矩阵。
    
    Args:
        feat (torch.Tensor): (N, d) 节点特征矩阵，d 为原始特征维度
        n_components (int): 降维后的目标维度，默认2
        normalization (str): 归一化类型，可选 'none'、'sym' 或 'rw'
    
    Returns:
        sparse_adj (torch.sparse_coo_tensor): (N, N) 归一化后的稀疏邻接矩阵（Laplacian形式）
    """
    n_components=2
    device = feat.device 
    num_nodes = feat.shape[0]

    # 降维：如果 feat 的维度大于 n_components，则使用 UMAP 降维
    feat_np = feat.cpu().numpy()
    reducer = umap.UMAP(n_components=n_components)
    reduced_data_np = reducer.fit_transform(feat_np)
    
    # 利用降维后的数据构造 Delaunay 三角剖分
    positions = reduced_data_np
    delaunay = Delaunay(positions, qhull_options='QJ')
    edges = set()  # 用 set 去重
    for simplex in delaunay.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add((simplex[i], simplex[j]))
                edges.add((simplex[j], simplex[i]))  # 无向图，双向边

    # 构造 edge_index 和 edge_weight（初始边权均为1）
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous().to(device)  # (2, num_edges)
    edge_weight = torch.ones(edge_index.shape[1], device=device)

    # 调用已有的 get_sparse_laplacian 处理边权（例如归一化）
    edge_index, edge_weight = get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization=norm_type)

    # 构造稀疏邻接矩阵（若 normalization 不为 'none'，则为 Laplacian 形式）
    sparse_adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes,num_nodes))
    return sparse_adj

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph

from sklearn import metrics
from munkres import Munkres

EOS = 1e-10


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def edge_deletion(adj, drop_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    samples = np.random.choice(num_edge, size=int(drop_r * num_edge), replace=False)
    dropped_edge_index = half_edge_index[:, samples].T
    adj[dropped_edge_index[:,0],dropped_edge_index[:,1]] = 0.
    adj[dropped_edge_index[:,1],dropped_edge_index[:,0]] = 0.
    return adj

def edge_addition(adj, add_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    num_node = adj.shape[0]
    added_edge_index_in = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    added_edge_index_out = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    adj[added_edge_index_in,added_edge_index_out] = 1.
    adj[added_edge_index_out,added_edge_index_in] = 1.
    return adj


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.cuda()


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]


def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def nearest_neighbors_sparse(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0])
    [s_, d_, val] = sp.find(adj)
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))
    return s, d


def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + EOS)
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse_coo_tensor(adj.indices(), new_values, adj.size()).coalesce()


def dgl_graph_to_torch_sparse(dgl_graph):
    # 获取边的索引 (source, destination)
    src, dst = dgl_graph.edges()
    
    # 获取边的权重
    if 'w' in dgl_graph.edata:
        values = dgl_graph.edata['w']
    else:
        values = torch.ones(len(src), device=dgl_graph.device)  # 如果无权重，默认全1

    # 构造索引
    indices = torch.stack([src, dst])

    # 构造 sparse tensor
    sparse_mx = torch.sparse_coo_tensor(indices, values, (dgl_graph.num_nodes(), dgl_graph.num_nodes()),device=values.device)
    
    return sparse_mx

def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


# def torch_sparse_to_dgl_graph(torch_sparse_mx):
#     torch_sparse_mx = torch_sparse_mx.coalesce()
#     indices = torch_sparse_mx.indices()
#     values = torch_sparse_mx.values()
#     rows_, cols_ = indices[0,:], indices[1,:]
#     dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
#     dgl_graph.edata['w'] = values.detach().cuda()
#     return dgl_graph


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse_coo_tensor(indices, values)
    return torch_sparse_mx


def remove_self_loop(adjs):
    adjs_ = []
    for i in range(len(adjs)):
        adj = adjs[i].coalesce()
        non_diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
        adj = torch.sparse_coo_tensor(adj.indices()[:, non_diag_index], adj.values()[non_diag_index], adj.shape).coalesce()
        adjs_.append(adj)
    return adjs_


def get_sparse_diag(adj):
    adj = adj.coalesce()
    diag_index = torch.nonzero(adj.indices()[0] == adj.indices()[1]).flatten()
    values = adj.values()[diag_index]
    indices = adj.indices()[:, diag_index]
    return indices, values


def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse_coo_tensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new.coalesce()

def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse_coo_tensor(indices, values)

def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return torch.sparse_coo_tensor(index, torch.ones(len(index[0])).to(adj.device), adj.shape).coalesce()


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print(numclass1, numclass2)
            print(self.pred_label)
            print(self.pred_label.max(), self.pred_label.min())
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, print_results=True):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        if print_results:
            print('ACC={:.4f}, f1_macro={:.4f}, precision_macro={:.4f}, recall_macro={:.4f}, f1_micro={:.4f}, '
                  .format(acc, f1_macro, precision_macro, recall_macro, f1_micro) +
                  'precision_micro={:.4f}, recall_micro={:.4f}, NMI={:.4f}, ADJ_RAND_SCORE={:.4f}'
                  .format(precision_micro, recall_micro, nmi, adjscore))

        return acc, nmi, f1_macro, adjscore
    
    




import sys
import os
import yaml
import random
import numpy as np
import torch
from typing import Optional

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes


def edgeidx2sparse(edge_index, num_nodes, edge_attr=None):
     sparse_edge_index = SparseTensor.from_edge_index(
         edge_index, sparse_sizes=(num_nodes, num_nodes)
     ).to(edge_index.device)
     sparse_edge_index.set_value_(edge_attr, layout="coo")
     return sparse_edge_index


def softmax_with_temp(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
    t: float = 1.
) -> Tensor:
    assert t > 0
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce='max')
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = ((src - src_max) / t).exp()
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        out = src - src_max.index_select(dim, index)
        out = (out / t).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / out_sum


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_config(args, path, mode):

    config_file = os.path.join(path, f'{args.dataset}.yml')
    with open(config_file, "r") as f:
        conf = yaml.load(f, yaml.FullLoader)
    conf = conf[mode]
    
    float_set = {
        "decoder_dropout", "encoder_dropout", "lr", "temp", "weight_decay", "weight_decay_prob"
    }
    for k, v in conf.items():
        if k in float_set:
            v = float(v)
        setattr(args, k, v)

    return args

def max_pool_fusion(image_original_adj,text_original_adj,device):
        image_adj = image_original_adj.coalesce()
        text_adj = text_original_adj.coalesce()

        image_indices = image_adj.indices().to(device)
        image_values = image_adj.values().to(device)
        text_indices = text_adj.indices().to(device)
        text_values = text_adj.values().to(device)

        combined_indices = torch.cat((image_indices, text_indices), dim=1)
        combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)

        combined_values_image = torch.full((combined_indices.size(1),), float('-inf')).to(device)
        combined_values_text = torch.full((combined_indices.size(1),), float('-inf')).to(device)

        combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
        combined_values_text[unique_idx[image_indices.size(1):]] = text_values
        combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)

        fusion_adj = torch.sparse.FloatTensor(combined_indices, combined_values, image_adj.size()).coalesce()

        return fusion_adj

def print_desc(bar, run, loss, result_dict, monitor=None, best_valid=None, best_epoch=None, prefix="Pre-training"):
    desc = f"[{prefix} #{run + 1:02d}] loss = {loss:.4f}, {'/'.join(result_dict.keys())}: "
    val, test = [], []
    for key, res in result_dict.items():
        val.append(f"{res[0]:.2%}" if res is not None else "NA")
        test.append(f"{res[1]:.2%}" if res is not None else "NA")
    desc += f"val = {'/'.join(val)}, test = {'/'.join(test)}"
    if best_valid is not None:
        desc += f", best valid {monitor} = {best_valid:.2%} at ep {best_epoch:02d}"
    bar.set_description(desc)


class Logger(object):
    def __init__(self, name, runs, time, info=None, log_path=None):
        self.name = name
        self.info = info
        self.results = [[] for _ in range(runs)]
        if log_path is not None and not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_path = log_path
        self.time = time

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, f=sys.stdout, last_best=False, print_info=False, percentage=True):
        result = (100 if percentage else 1) * torch.tensor(self.results)

        best_results = []

        for r in result:
            valid = r[:, 0].max().item()
            if last_best:
                # get last max value index by reversing result tensor
                argmax = r.size(0) - r[:, 0].flip(dims=[0]).argmax().item() - 1
            else:
                argmax = r[:, 0].argmax().item()
            test = r[argmax, 1].item()
            best_results.append((valid, test))

        best_result = torch.tensor(best_results)
        print(f' Final {self.name}(%): val = {best_result[:, 0].mean():.2f} ± {best_result[:, 0].std():.2f}, '
              f'test = {best_result[:, 1].mean():.2f} ± {best_result[:, 1].std():.2f}', file=f)

        if self.log_path is not None:
            with open(os.path.join(self.log_path, 'log.txt'), 'a') as log:
                log.write(f'[{self.time}]{self.info}:\n' if print_info else '')
                log.write(f'({self.name}): val = {best_result[:, 0].mean():.2f} ± {best_result[:, 0].std():.2f}\t'
                          f'test = {best_result[:, 1].mean():.2f} ± {best_result[:, 1].std():.2f}\n')
