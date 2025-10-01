


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
   
        self.scale = embed_dim ** 0.5

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.gating = nn.Parameter(torch.ones(1))  

    def forward(self, text_embedding, vision_embedding, mask=None):
        """
        text_embedding: (batch_size, seq_len, embed_dim) -> Q
        vision_embedding: (batch_size, num_images, embed_dim) -> K, V
        mask: (batch_size, seq_len, num_images), 1 代表允许跨模态交互，0 代表屏蔽
        """
        Q = self.query_proj(text_embedding)  # 计算 Query
        K = self.key_proj(vision_embedding)  # 计算 Key
        V = self.value_proj(vision_embedding)  # 计算 Value

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # QK^T / sqrt(d)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))  # 应用 Mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)  # 计算 Softmax
        attn_probs = self.attn_dropout(attn_probs)

        attended_features = torch.matmul(attn_probs, V)  # (B, seq_len, embed_dim)

        # 门控机制 & 残差连接
        fused_features = self.gating * attended_features + text_embedding
        return self.out_proj(fused_features)
    
    
class LinearFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        # 把 (d_v + d_t) -> d_out
        self.linear = nn.Linear(d*2, d)

    def forward(self, v_feat, t_feat):
        # v_feat: [N, d_v]
        # t_feat: [N, d_t]
        fused = torch.cat([v_feat, t_feat], dim=1)  
        out = self.linear(fused)                   
        out = F.relu(out)                          
        return out



class GatingFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        # 门控网络把 d -> d, 输出一个向量
        self.gate = nn.Linear(d, d)
    
    def forward(self, v_feat, t_feat):
        mid = (v_feat + t_feat) / 2.0
        alpha = torch.sigmoid(self.gate(mid))  # shape=[N, d]
        
        # 最终: fused = alpha * v_feat + (1-alpha)* t_feat
        fused = alpha * v_feat + (1 - alpha) * t_feat
        return fused
    