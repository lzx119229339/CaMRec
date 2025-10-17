
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softplus
import torch.nn.functional as F

def sce_loss(self,x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss


def kl_divergence(mu_p, mu_q):
    diff = mu_q - mu_p
    squared_diff = torch.sum(diff * diff, dim=-1)  # Sum along the last dimension (assuming mu_p and mu_q are batched)    
    return squared_diff.mean()  # Take the mean of the KL divergence across the batch and extract the scalar value

def kl_vmf(loc_p, loc_q):
    loc_p = loc_p / loc_p.norm(dim=-1, keepdim=True)
    loc_q = loc_q / loc_q.norm(dim=-1, keepdim=True)
    return -(loc_p * loc_q).sum(-1).mean()

def ortho_loss(z1, zs, norm=True, temp=0.1):
    z1 = F.normalize(z1, dim=-1)
    zs = F.normalize(zs, dim=-1)
    if norm:
        return torch.norm(torch.matmul(z1.T, zs)) # yes (type1)
    else:
        raise NotImplementedError('Please set norm=True')

def ortho_loss_focal(z1, zs):
    assert z1.shape == zs.shape
    z1 = F.normalize(z1, dim=-1)
    zs = F.normalize(zs, dim=-1)
    return torch.matmul(z1, zs.T).diag().mean()
def orthogonal_loss(embeddings,index):
    """ 让 embeddings 彼此正交 """
    embeddings = embeddings[index]
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # 归一化
    sim_matrix = torch.mm(norm_emb, norm_emb.T)  # 计算相似度
    identity = torch.eye(sim_matrix.shape[0]).to(embeddings.device)
    loss = torch.norm(sim_matrix - identity)  # 使非对角线项接近0
    return loss