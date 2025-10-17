import dgl.function as fn
import torch
import torch.nn as nn

class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()

    def forward(self,x,g):
        with g.local_scope():
            g.ndata['h'] = x
            degs = g.in_degrees().float().clamp(min=1)  
            norm = torch.pow(degs, -0.5).unsqueeze(1).to(x.device)
            g.ndata['h'] = g.ndata['h'] * norm
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h')) 
            g.ndata['h'] = g.ndata['h'] * norm  
            return g.ndata['h']
        
        

class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index,size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size=None):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)



class GCN_dgl(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(GCN_dgl, self).__init__()
        self.layers = nn.ModuleList()

      
        self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
        self.layers.append(GCNConv_dgl(hidden_channels, out_channels))

        self.dropout = dropout

        self.dropout_adj = nn.Dropout(p=dropout_adj)


    def forward(self, x,Adj_1):
    
        Adj = copy.deepcopy(Adj_1).to(x.device)
        Adj.edata['w'] = self.dropout_adj(Adj.edata['w'])

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x
    
class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']