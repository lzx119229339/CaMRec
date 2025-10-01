# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation
# Update: 01/08/2022
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian

from models.Unet import UNet
from models.diffusion_ver9 import diffusion

class freedom_mcdrec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(freedom_mcdrec, self).__init__(config, dataset)
        self.config = config

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.dropout_prob = 0.1
        self.degree_ratio = config['degree_ratio']
        self.w = config['w']
#         self.w_dm = config['w_dm']
        self.temperature = config['temperature']

        self.n_nodes = self.n_users + self.n_items
        
        # diffusion
        self.diff_weight = config['diff_weight'] 
        self.model = UNet(self.config)
        self.diff = diffusion(self.config)
        self.steps = config['timesteps']
        

        
        
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None


        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.sample_x = None
        
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
            
            
        self.mlp_1 = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(64, self.feat_embed_dim)
        )

        nn.init.xavier_normal_(self.mlp_1[0].weight)  # 初始化第一个线性层
        nn.init.xavier_normal_(self.mlp_1[3].weight)  # 初始化第二个线性层
        nn.init.xavier_normal_(self.mlp_1[6].weight)  # 初始化最后一个线性层

            
            

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self, epoch_idx):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        
#         ipdb.set_trace()
        # degree-sensitive edge pruning
    
        edge_indices, edge_values = self.get_edge_info(epoch_idx)
        edge_indices, edge_values = edge_indices.to(self.device), edge_values.to(self.device)
        edge_full_indices = torch.arange(edge_values.size(0)).to(self.device)
        
        degree_len = int(edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(edge_values, degree_len)
        # random sample
        keep_indices = edge_indices[:, degree_idx]
        
        
        # norm values
        keep_values = self._normalize_adj_m_dm(keep_indices, torch.Size((self.n_users, self.n_items)), epoch_idx)
#         keep_values = keep_values + edge_scores
        
        all_values = torch.cat((keep_values, keep_values))
        
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m_dm(self, indices, adj_size, epoch_idx):
#         ipdb.set_trace()
#         adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        with torch.no_grad():
            if epoch_idx ==0:
                item_embs = self.item_id_embedding.weight
            else:
                item_embs = self.sample_x
            user_embs = torch.nn.functional.normalize(self.user_embedding.weight.detach(), p=2, dim=-1)
            item_embs = torch.nn.functional.normalize(item_embs.detach(), p=2, dim=-1)

            scores = torch.matmul(user_embs, item_embs.transpose(0, 1))
            edge_scores = scores[indices[0], indices[1]]
            edge_scores = edge_scores * self.temperature

        
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]) + edge_scores.detach(), adj_size)
        
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values
    
    
    def _normalize_adj_m(self, indices, adj_size):
#         ipdb.set_trace()
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self, epoch_idx):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m_dm(edges.cuda(), torch.Size((self.n_users, self.n_items)), epoch_idx)
        return edges, values


    def forward(self, adj, predicted_x):
#         ipdb.set_trace()

        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)

        h = self.item_id_embedding.weight
        
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

            
        i_inputs = self.w * predicted_x + (1-self.w) * self.item_id_embedding.weight
        ego_embeddings = torch.cat((self.user_embedding.weight, i_inputs), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h 

    def bpr_loss(self, users, pos_items, neg_items):
#         ipdb.set_trace()
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            
#         ipdb.set_trace()
            
        t= torch.randint(low=0, high=self.steps, size=(text_feats.shape[0] // 2 + 1,)).cuda()
        t = torch.cat([t, self.steps - t - 1], dim=0)[:text_feats.shape[0]]
        
            


        diff_loss, predicted_x = self.diff.p_losses(self.model, self.item_id_embedding.weight, text_feats, image_feats, t, noise=None, loss_type="l2") 
        
#         diff_loss, predicted_x, predicted_t, predicted_v  = self.diff.p_losses(self.model, self.item_id_embedding.weight, text_feats, image_feats, t, noise=None, loss_type="l2")     
            
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj, predicted_x)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)
        
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
        return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss) + self.diff_weight * diff_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj, self.sample_x)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    
    def sample(self):
#         ipdb.set_trace()
#         user = interaction[0]  # 4096
        text_feats, image_feats = None, None
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            


        predicted_x = self.diff.sample(self.model, self.item_id_embedding.weight, text_feats, image_feats)
#         ipdb.set_trace()
        self.sample_x = predicted_x.detach()
