import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import scipy
import numpy as np

from utils import *
from IOUtils import *

'''
GCN
'''
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support) # 稀疏矩阵的矩阵乘法
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class TPTGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0):
        super(TPTGCN, self).__init__()

        self.nfeat = nfeat

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.linear1 = nn.Linear(nfeat, nhid)

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(nhid, nhid))
        self.bias_gate = nn.Parameter(torch.FloatTensor(nhid))

        self.gc2 = GraphConvolution(nhid, nclass)
        self.linear2 = nn.Linear(nhid, nclass)

        self.dropout = dropout

        self.init_parameters()

    def init_parameters(self):
        """
        Initializing weights.
        """
        # torch.nn.init.xavier_uniform_(self.weight_matrix)
        init_range = np.sqrt(6.0 / (self.nfeat + self.nfeat))
        torch.nn.init.uniform_(self.weight_matrix, -init_range, init_range)
        torch.nn.init.constant_(self.bias_gate, 0)

    def highway(self, layer1, layer2):
        transform_gate = torch.mm(layer1, self.weight_matrix) + self.bias_gate
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = torch.tensor(1, dtype=torch.float32) - transform_gate
        highway_x = transform_gate * layer2 + carry_gate * layer1

        return highway_x


    def forward(self, e_x, r_x, prim_adj, rela_adj):
        # x = F.relu(self.gc1(x, adj))
        gcn_x = torch.relu(self.gc1(e_x, prim_adj))
        # transform_gate = torch.mm(x, self.weight_matrix) + self.bias_gate
        # transform_gate = torch.sigmoid(transform_gate)
        # carry_gate = torch.tensor(1, dtype=torch.float32) - transform_gate
        # x = transform_gate * gcn_x + carry_gate * x

        # gcn_layer_output = torch.cat((gcn_x, r_x), 0)
        x = self.highway(e_x, gcn_x)
        gcn_layer_input = torch.cat((x, r_x), 0)
        x = F.dropout(gcn_layer_input, self.dropout, training=self.training)
        gcn_x = torch.relu(self.gc2(x, rela_adj)) # 这里RDGCN用了激活函数，原始的gcn没有用
        x = self.highway(x, gcn_x)

        # return F.log_softmax(x, dim=1)
        return x


'''
GCN variants
'''
class DualityGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0):
        super(DualityGCN, self).__init__()

        self.nfeat = nfeat

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.linear1 = nn.Linear(nfeat, nhid)

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(nhid, nhid))
        self.bias_gate = nn.Parameter(torch.FloatTensor(nhid))

        self.gc2 = GraphConvolution(nhid, nclass)
        self.linear2 = nn.Linear(nhid, nclass)

        self.dropout = dropout

        self.init_parameters()

    def init_parameters(self):
        """
        Initializing weights.
        """
        # torch.nn.init.xavier_uniform_(self.weight_matrix)
        init_range = np.sqrt(6.0 / (self.nfeat + self.nfeat))
        torch.nn.init.uniform_(self.weight_matrix, -init_range, init_range)
        torch.nn.init.constant_(self.bias_gate, 0)

    def highway(self, layer1, layer2):
        transform_gate = torch.mm(layer1, self.weight_matrix) + self.bias_gate
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = torch.tensor(1, dtype=torch.float32) - transform_gate
        highway_x = transform_gate * layer2 + carry_gate * layer1

        return highway_x


    def forward(self, e_x, r_x, prim_adj, rela_adj):
        # x = F.relu(self.gc1(x, adj))
        gcn_x = torch.relu(self.gc1(e_x, prim_adj))
        # transform_gate = torch.mm(x, self.weight_matrix) + self.bias_gate
        # transform_gate = torch.sigmoid(transform_gate)
        # carry_gate = torch.tensor(1, dtype=torch.float32) - transform_gate
        # x = transform_gate * gcn_x + carry_gate * x

        # gcn_layer_output = torch.cat((gcn_x, r_x), 0)
        x = self.highway(e_x, gcn_x)
        gcn_layer_input = torch.cat((x, r_x), 0)
        x = F.dropout(gcn_layer_input, self.dropout, training=self.training)
        gcn_x = torch.relu(self.gc2(x, rela_adj)) # 这里RDGCN用了激活函数，原始的gcn没有用
        x = self.highway(x, gcn_x)

        # return F.log_softmax(x, dim=1)
        return x

class HeterRelGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0):
        super(HeterRelGCN, self).__init__()

        self.nfeat = nfeat

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.linear1 = nn.Linear(nfeat, nhid)

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(nhid, nhid))
        self.bias_gate = nn.Parameter(torch.FloatTensor(nhid))

        self.gc2 = GraphConvolution(nhid, nclass)
        self.linear2 = nn.Linear(nhid, nclass)

        self.dropout = dropout

        self.init_parameters()

    def init_parameters(self):
        """
        Initializing weights.
        """
        # torch.nn.init.xavier_uniform_(self.weight_matrix)
        init_range = np.sqrt(6.0 / (self.nfeat + self.nfeat))
        torch.nn.init.uniform_(self.weight_matrix, -init_range, init_range)
        torch.nn.init.constant_(self.bias_gate, 0)

    def highway(self, layer1, layer2):
        transform_gate = torch.mm(layer1, self.weight_matrix) + self.bias_gate
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = torch.tensor(1, dtype=torch.float32) - transform_gate
        highway_x = transform_gate * layer2 + carry_gate * layer1
        return highway_x


    def forward(self, e_x, r_x, heter_adj):
        # x = F.relu(self.gc1(x, adj))
        gcn_layer_input = torch.cat((e_x, r_x), 0)
        gcn_x1 = torch.relu(self.gc1(gcn_layer_input, heter_adj))
        # transform_gate = torch.mm(x, self.weight_matrix) + self.bias_gate
        # transform_gate = torch.sigmoid(transform_gate)
        # carry_gate = torch.tensor(1, dtype=torch.float32) - transform_gate
        # x = transform_gate * gcn_x + carry_gate * x

        # gcn_layer_output = torch.cat((gcn_x, r_x), 0)
        x = self.highway(gcn_layer_input, gcn_x1)
        x = F.dropout(x, self.dropout, training=self.training)
        gcn_x = torch.relu(self.gc2(x, heter_adj)) # 这里RDGCN用了激活函数，原始的gcn没有用
        x = self.highway(x, gcn_x)

        # return F.log_softmax(x, dim=1)
        return x


'''
Attention
'''
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        this_q = q[0, -1, :]
        residual = this_q

        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        q = self.w_qs(this_q).view(sz_b, 1, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, 1, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, 1, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner,
            dropout=dropout)

    def forward(self, enc_input, enc_proximity):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_proximity, enc_proximity)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__( self, e_dim, n_layers, n_head, d_k, d_v, dropout=0.1):

        super().__init__()

        self.entity_dim = e_dim

        self.relation_dim = e_dim
        self.positional_weight = nn.Parameter(torch.Tensor(self.relation_dim, self.entity_dim))

        self.init_parameters()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(e_dim, e_dim, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def init_parameters(self):
        """
        Initializing weights.
        """
        nn.init.xavier_uniform_(self.positional_weight)

    def forward(self, rel_seq, head_seq, kernel_em, e_features, r_features, return_attns=False):

        enc_slf_attn_list = []

        relation_em = torch.index_select(r_features, 0, rel_seq)
        relation_em = torch.mm(relation_em, self.positional_weight)
        entity_em = torch.index_select(e_features, 0, head_seq)

        '''
        这一步能否用NTN(2019-10-28 10:42:25)
        '''
        # seq_input_em = relation_em + entity_em
        seq_input_em = torch.mul(relation_em, entity_em) # 改为elementwise multiplication

        kernel_em = kernel_em.unsqueeze(0).unsqueeze(0)
        enc_output = seq_input_em.unsqueeze(0)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(kernel_em, enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        entity_enc_em = enc_output[0, -1, :]

        return entity_enc_em