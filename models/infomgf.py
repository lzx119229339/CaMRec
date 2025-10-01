
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric
import math
import sys 
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from utils.utils import * 


class INFOMGF(GeneralRecommender):
    def __init__(self, config, dataset):
        super(INFOMGF, self).__init__(config, dataset)
   
        self.sparse = True
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.num_modal = 2
        self.dropout =config['dropout']
        self.n_mm_layers = config['n_mm_layers']
        self.reg_weight = config['reg_weight']
        self.weight_mi = config['weight_mi']
        self.n_ui_layers = config['n_ui_layers']
        
        
       

        
        ## Load Initial Embedding 
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        
        self.item_id_feat= nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        self.item_image_feat = nn.Embedding.from_pretrained(self.v_feat, freeze=False).to(self.device)
        self.item_text_feat = nn.Embedding.from_pretrained(self.t_feat, freeze=False) .to(self.device)
        
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim).to(self.device)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)

        nn.init.xavier_uniform_(self.item_id_feat.weight)
        nn.init.xavier_normal_(self.image_trs.weight)
        nn.init.xavier_normal_(self.text_trs.weight)
        
        ## Load Graph 
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))

        if os.path.exists(image_adj_file):
            self.image_adj = torch.load(image_adj_file)
        else:
            image_adj = build_sim(self.item_image_feat.weight.detach())
            self.image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
                                                    norm_type='sym')
            torch.save(self.image_adj, image_adj_file)
    
        if os.path.exists(text_adj_file):
                self.text_adj = torch.load(text_adj_file)
        else:
            text_adj = build_sim(self.item_text_feat.weight.detach())
            self.text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')
            torch.save(self.text_adj, text_adj_file)
   
        
        with torch.no_grad():
            self.image_adj = self.image_adj.to(self.device)
            self.text_adj = self.text_adj.to(self.device)
        self.original_adjs = []
        self.original_adjs.append(self.text_adj)
        self.original_adjs.append(self.image_adj)
        
        edge_index = torch.tensor(self.pack_edge_index(self.interaction_matrix), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1).to(self.device)
       
        ## Spectrum Learning 
        self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_f = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=-1)
        # self.query_v = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        # )
        # self.query_t = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        # )
        # self.gate_image_prefer = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.Sigmoid()
        # )

        # self.gate_text_prefer = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.Sigmoid()
        # )
        # self.gate_fusion_prefer = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.Sigmoid()
        # )
        
        ## Graph Learner 
        self.k_graphlearner = config['k_graphlearner']
        self.dropedge_rate = config['dropedge_rate']
        self.activation_learner = config['activation_learner']
        self.uni_graph_learner = [ATT_learner(2, self.feat_embed_dim, self.k_graphlearner, 6, self.dropedge_rate,  self.activation_learner).to(self.device) for _ in range(self.num_modal)]
        self.mm_graph_learner = ATT_learner(2, self.feat_embed_dim, self.k_graphlearner, 6, self.dropedge_rate, self.activation_learner).to(self.device)

        ## Augmentation Graph Learner 
       
        self.hidden_dim = config['hidden_dim']
        self.aug_lambda = config['aug_lambda']

        self.uni_auggraph_learner = AugGraphGenerator(self.embedding_dim, self.hidden_dim, self.embedding_dim, self.aug_lambda, self.dropout, self.dropedge_rate).to(self.device)
        
        
        
       
        ## Graph Encoder  and Contrastive Learning Projection
      
        # self.encoder = GraphEncoder(self.n_mm_layers, self.feat_embed_dim, self.hidden_dim, self.feat_embed_dim, self.dropout).to(self.device)
        
        ## Mutual Information 
        # self.proj_dim = config['proj_dim']
        
        # self.proj_uni = nn.ModuleList([nn.Sequential(
        #     nn.Linear(self.feat_embed_dim, self.proj_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.proj_dim, self.proj_dim)
        # ).to(self.device) for _ in range(self.num_modal)])
        
        # self.proj_mm = nn.Sequential(
        #     nn.Linear(self.feat_embed_dim, self.proj_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.proj_dim, self.proj_dim)
        # ).to(self.device)
        
        # self.proj_aug = nn.ModuleList([nn.Sequential(
        #     nn.Linear(self.feat_embed_dim, self.proj_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.proj_dim, self.proj_dim)
        # ).to(self.device) for _ in range(self.num_modal)])
        
        
        

    def forward(self, train=False):
  

        item_image_embedding = self.image_trs(self.item_image_feat.weight)
        item_text_embedding = self.text_trs(self.item_text_feat.weight)
        
        image_conv,text_conv,fusion_conv = self.spectrum_convolution(item_image_embedding,item_text_embedding)
        item_image_embeds = torch.multiply(self.item_id_feat.weight,self.gate_v(image_conv))
        item_text_embeds = torch.multiply(self.item_id_feat.weight,self.gate_t(text_conv))
        fusion_item_embeds = torch.multiply(self.item_id_feat.weight,self.gate_f(fusion_conv))
        
        item_embeds = self.item_id_feat.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
    
        z_id_user,z_id_item = torch.split(all_embeddings,[self.n_users,self.n_items],dim=0)
        
     
        
        ## Item-Item Modal Graph Learning 
        #### uni Graph Learning 
        feat_list = []
        feat_list.append(item_text_embeds)
        feat_list.append(item_image_embeds)
        #### 先做邻域聚合 
        embed_list = self.aggregate(feat_list,self.original_adjs,self.n_mm_layers)
        #### 然后学习模态特定的图 
        ##### encoder可以换成每个模态特定的  
        uni_graph_list = [ ]
        for i in range(self.num_modal):
            uni_embedding = self.uni_graph_learner[i](embed_list[i])
            uni_graph = self.uni_graph_learner[i].graph_process(uni_embedding)
            uni_graph_list.append(uni_graph)
        #### Multi Modal Graph Learning 
        mm_embedding = self.mm_graph_learner(fusion_item_embeds)
        mm_graph = self.mm_graph_learner.graph_process(mm_embedding)
        #### Augment Graph Learning 
        auggraph_list = self.graph_generative_augment(self.original_adjs,feat_list)
        #### Encoding 
        z_aug = self.aggregate(feat_list,auggraph_list,self.n_mm_layers)
        
        z_uni = self.aggregate(feat_list,uni_graph_list,self.n_mm_layers)
        mm_graph = dgl_graph_to_torch_sparse(mm_graph).to(self.device)
        z_item_mm = torch.sparse.mm(mm_graph,fusion_item_embeds)
        # z_item_mm = self.encoder(fusion_item_embeds,mm_graph)

        #### Residual 
        h_item = z_item_mm + z_id_item 
        
        ## User View Learning 
        

        z_item_text,z_item_image = z_uni
        z_item_text_aug,z_item_image_aug = z_aug
        
        z_user_text = torch.sparse.mm(self.R,z_item_text)
        z_user_image = torch.sparse.mm(self.R,z_item_image)
        z_user_text_aug = torch.sparse.mm(self.R,z_item_text_aug)
        z_user_image_aug = torch.sparse.mm(self.R,z_item_image_aug)
        z_user_mm = torch.sparse.mm(self.R,z_item_mm)
        
        h_user= z_user_mm + z_id_user
        
        z_uni_user = []
        z_uni_user.append(z_user_text)
        z_uni_user.append(z_user_image)
        z_aug_user = [] 
        z_aug_user.append(z_user_text_aug)
        z_aug_user.append(z_user_image_aug)
        
        

        if train: 
            return h_user,h_item,z_user_mm,z_uni_user,z_aug_user,z_item_mm,z_uni,z_aug,embed_list,z_id_user
        else: 
            return h_user,h_item
       

  
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
    
    def InfoNCE(self, view1, view2, temperature=0.2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
    
    def mutual_information_loss(self, z_mm,z_uni,z_aug,index=None):

        if index is None:
            index = torch.arange(z_mm.shape[0], device=self.device)  # 默认选取所有节点

        num_nodes, _ = z_mm[index].size()
        z_proj_uni = [z_uni[i][index] for i in range(self.num_modal)]
        z_proj_aug = [z_aug[i][index] for i in range(self.num_modal)]
        z_proj_mm = z_mm[index]
        

        loss_smi = 0
        loss_smi += self.InfoNCE(z_proj_uni[0], z_proj_uni[1])
    
     

        loss_fused = 0
        loss_umi = 0
        loss_fused += self.InfoNCE(z_proj_mm,z_proj_uni[0])
        loss_fused += self.InfoNCE(z_proj_mm,z_proj_uni[1])
        loss_umi += self.InfoNCE(z_proj_uni[0], z_proj_aug[0])
        loss_umi += self.InfoNCE(z_proj_uni[1], z_proj_aug[1])



        loss = loss_fused + loss_smi + loss_umi
        return loss
    
    def calculate_loss(self,interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
      
      
        h_user,h_item,z_user_mm,z_uni_user,z_aug_user,z_item_mm,z_uni,z_aug,embed_list= self.forward(train=True)
       
        ## loss bpr 
        u_g_embeddings = h_user[users]
        pos_i_g_embeddings = h_item[pos_items]
        neg_i_g_embeddings = h_item[neg_items]
        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)
        loss_bpr = batch_mf_loss + batch_emb_loss + batch_reg_loss
        ## mutual information loss 
        loss_mi = self.mutual_information_loss(z_mm=h_user,z_uni=z_uni_user,z_aug=z_aug_user,index=users)+self.mutual_information_loss(z_mm=h_item,z_uni=z_uni,z_aug=z_aug,index=pos_items)
        ## GenerativeLoss 
        loss_gen = self.uni_auggraph_learner.cal_loss_dis(z_aug=z_aug,z_uni=z_uni,ori_feats=embed_list,index=pos_items)+self.uni_auggraph_learner
        import ipdb; ipdb.set_trace()
        loss = loss_bpr + self.weight_mi * loss_mi  + loss_gen 
    
      
        return loss
    
        

    def full_sort_predict(self, interaction):
        user = interaction[0]
        z_mm_user,z_mm_item = self.forward(train=False)
        u_embeddings = z_mm_user[user]
        scores = torch.matmul(u_embeddings,z_mm_item.transpose(0,1))
        return scores 
        
    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            rowsum[rowsum == 0] = 1  # 避免除0错误
            
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def aggregate(self,feat_list,adjs,num_layers):
 
        embed_list = [] 
        for i in range(self.num_modal):
            z =feat_list[i]
            adj = adjs[i]
            if isinstance(adj,dgl.DGLGraph):
                adj = dgl_graph_to_torch_sparse(adj).to(self.device)
            
            for i in range(num_layers):
                z = torch.sparse.mm(adj,z)
            embed_list.append(z)
        return embed_list
        
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        pass
        # self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        # self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))
    
    def graph_generative_augment(self,adjs,feat_list):
   
        adjs_aug = []
        adjs = remove_self_loop(adjs)
        for i in range(self.num_modal):

            edge_index = adjs[i].indices()
            adj_aug_value = self.uni_auggraph_learner(feat_list[i],edge_index)
            adj_aug = torch.sparse.FloatTensor(edge_index, adj_aug_value, (feat_list[i].shape[0], feat_list[i].shape[0])).to(self.device)
            adj_aug = normalize_adj(adj_aug,'sym')
            adjs_aug.append(adj_aug)
        
      
        adjs_aug = [torch_sparse_to_dgl_graph(a) for a in adjs_aug]
        
        return adjs_aug
    def spectrum_convolution(self, image_embeds, text_embeds):
        """
        Modality Denoising & Cross-Modality Fusion
        """
        image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')           
        text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')

        image_complex_weight = torch.view_as_complex(self.image_complex_weight)   
        text_complex_weight = torch.view_as_complex(self.text_complex_weight)
        fusion_complex_weight = torch.view_as_complex(self.fusion_complex_weight)

        #   Uni-modal Denoising
        image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')    
        text_conv = torch.fft.irfft(text_fft * text_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

        #   Cross-modality fusion
        fusion_conv = torch.fft.irfft(text_fft * image_fft * fusion_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho') 
        
        return image_conv, text_conv, fusion_conv
    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

    
        



class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre

import dgl
import math


class ATT_learner(nn.Module):
    def __init__(self, nlayers, isize, k, i, dropedge_rate,  act):
        super(ATT_learner, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentive(isize))

        self.k = k
        self.non_linearity = 'relu'
        self.i = i
     
        self.act = act
        self.dropedge_rate = dropedge_rate

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)

        return h

    def forward(self, features):
        embeddings = self.internal_forward(features)

        return embeddings

    def graph_process(self, embeddings):
        
        rows, cols, values = knn_fast(embeddings, self.k, 1000)
        rows_ = torch.cat((rows, cols))
        cols_ = torch.cat((cols, rows))
        values_ = torch.cat((values, values))
        values_ = apply_non_linearity(values_, self.non_linearity, self.i)
        values_ = F.dropout(values_, p=self.dropedge_rate, training=self.training)
        learned_adj = dgl.graph((rows_, cols_), num_nodes=embeddings.shape[0], device='cuda')
        learned_adj.edata['w'] = values_
        return learned_adj
    

class Attentive(nn.Module):
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        return x @ torch.diag(self.w)
    
    

class AugGraphGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, rep_dim, aug_lambda, temperature=1.0, bias=0.0 + 0.0001):
        super(AugGraphGenerator, self).__init__()

        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, 1))

        self.temperature = temperature
        self.bias = bias
        self.aug_lambda = aug_lambda

        self.decoder = nn.Sequential(nn.Linear(rep_dim, input_dim))

    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h

    def get_edge_weight(self, embeddings, edges):
 
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2

    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(edges_weights_raw.device)
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        output = torch.sigmoid(gate_inputs).squeeze()

        return output

    def forward(self, embedding, edges):
        embedding_ = self.get_node_embedding(embedding)
        edges_weights_raw = self.get_edge_weight(embedding_, edges)
        weights = self.gumbel_sampling(edges_weights_raw)
        return weights

    def cal_loss_dis(self, z_uni,z_aug,ori_feats,index=None):
        if index is None:
            index = torch.arange(z_uni.shape[0], device=self.device)  # 默认选取所有节点

        loss_upmi = 0
        loss_rec = 0
        loss_upmi += self.aug_lambda * InfoNCE(z_aug[0][index],z_uni[0][index])
        loss_upmi += self.aug_lambda * InfoNCE(z_aug[1][index],z_uni[1][index])
        feat_agg_rec = self.decoder(z_aug[0])
        loss_rec += sce_loss(feat_agg_rec,ori_feats[0])
        feat_agg_rec = self.decoder(z_aug[1])
        loss_rec += sce_loss(feat_agg_rec,ori_feats[1])
        

         



        loss_dis = loss_upmi + loss_rec
        return loss_dis
    
    
def sce_loss(x, y, beta=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(beta)

    loss = loss.mean()
    return loss
def sim_con(z1, z2, temperature):
    z1_norm = F.normalize(z1,p=2,dim=-1)
    z2_norm = F.normalize(z2,p=2,dim=-1)
    sim_matrix = torch.mm(z1_norm,z2_norm.t())/temperature

    return sim_matrix
def calc_lower_bound(z_1, z_2, pos, temperature=0.2):
  
    EOS = 1e-8  # 避免数值错误
    sim_matrix = sim_con(z_1,z_2,temperature)
    matrix_1 = F.softmax(sim_matrix,dim=-1)
    matrix_2 = F.softmax(sim_matrix.t(),dim=-1)
    lori_1 = -torch.log((matrix_1.mul(pos).sum(dim=-1)) + EOS).mean()
    lori_2 = -torch.log((matrix_2.mul(pos).sum(dim=-1)) + EOS).mean()

    return (lori_1 + lori_2) / 2

def calc_upper_bound(z_1, z_2, pos, temperature=0.2):
    matrix_1 = sim_con(z_1, z_2, temperature)
    loss = matrix_1.mul(pos).sum(dim=-1).mean() - matrix_1.mean()

    return loss










class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.gnn_encoder_layers = nn.ModuleList()
        self.act = nn.ReLU()



        self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
        for _ in range(nlayers - 2):
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
        self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))

    def forward(self, x, Adj):
        # 对节点特征进行 dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 执行多层图卷积
        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        return x

import dgl.function as fn
class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']



import torch
import torch.nn as nn

class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadCrossModalAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性层
        self.query_fc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_fc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_fc = nn.Linear(embed_dim, embed_dim, bias=False)

        # 输出层
        self.fc_out = nn.Linear(embed_dim, embed_dim, bias=False)

        # 归一化 Softmax
        self.softmax = nn.Softmax(dim=-1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """使用 Xavier Normal 进行参数初始化"""
        nn.init.xavier_normal_(self.query_fc.weight)
        nn.init.xavier_normal_(self.key_fc.weight)
        nn.init.xavier_normal_(self.value_fc.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, feat_list):
        """
        :param feat_list: List of embeddings [id_emb, text_emb, image_emb]
                          每个 embedding 形状为 (num_nodes, embed_dim)
        :return: 融合后的 embedding (num_nodes, embed_dim)
        """
        id_emb, text_emb, image_emb = feat_list  # (num_nodes, embed_dim)
        num_nodes = id_emb.shape[0]

        # 计算 Query
        Q = self.query_fc(id_emb).view(num_nodes, self.num_heads, self.head_dim)  # (num_nodes, num_heads, head_dim)
        
        # 计算 Key 和 Value
        KV = torch.stack([id_emb, text_emb, image_emb], dim=1)  # (num_nodes, 3, embed_dim)
        K = self.key_fc(KV).reshape(num_nodes, 3, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (num_nodes, num_heads, 3, head_dim)
        V = self.value_fc(KV).reshape(num_nodes, 3, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (num_nodes, num_heads, 3, head_dim)

        # **修正 einsum 计算**
        attn_scores = torch.einsum("n h d, n h k d -> n h k", Q, K) / (self.head_dim ** 0.5)  # (num_nodes, num_heads, 3)
        attn_weights = self.softmax(attn_scores)  # (num_nodes, num_heads, 3)

        # 计算加权 Value
        fused_modalities = torch.einsum("n h k, n h k d -> n h d", attn_weights, V)  # (num_nodes, num_heads, head_dim)

        # 重新拼接回原始维度
        fused_emb = fused_modalities.reshape(num_nodes, -1)
        fused_emb = self.fc_out(fused_emb)  # 线性变换

        return fused_emb, attn_weights
    
def normalize_adj(adj, mode):
    """ 归一化稀疏邻接矩阵
    :param adj: torch.sparse.FloatTensor，形状 (num_nodes, num_nodes)
    :param mode: "sym" (对称归一化) 或 "row" (行归一化)
    :return: 归一化后的 torch.sparse.FloatTensor
    """
    adj = adj.coalesce()  # 确保是 coalesce 形式
    num_nodes = adj.size(0)

    # 计算度数
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()  # (num_nodes,)
    
    # **处理 0 度节点，避免 inf**
    row_sum = torch.where(row_sum <= 1e-8, torch.tensor(1.0, device=adj.device), row_sum)

    if mode == "sym":
        inv_sqrt_degree = torch.pow(row_sum, -0.5)  # 计算 D^(-1/2)
        inv_sqrt_degree[torch.isinf(inv_sqrt_degree)] = 0.0  # 避免 inf
        D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]  # (nnz,)
    
    elif mode == "row":
        inv_degree = 1.0 / row_sum  # 计算 D^(-1)
        D_value = inv_degree[adj.indices()[0]]  # (nnz,)

    else:
        raise ValueError(f"Unknown norm mode: {mode}")

    # **检查是否存在 NaN 或 inf**
    if torch.isnan(D_value).any() or torch.isinf(D_value).any():
        raise RuntimeError("normalize_adj: D_value contains NaN or Inf!")

    # 计算归一化邻接矩阵
    new_values = adj.values() * D_value
    norm_adj = torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()

    return norm_adj

def InfoNCE(view1, view2, temperature=0.2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)