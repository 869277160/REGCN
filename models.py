import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import scipy
import numpy as np

from utils import *
from layers import *
from IOUtils import *



class TradicGCN(nn.Module):
    def __init__(self, init_dim, eitity_cnt, device):
        super(TradicGCN, self).__init__()

        self.hidden_dim = 300
        self.output_dim = 300

        self.eitity_cnt = eitity_cnt

        self.neg_k = 125

        self.gamma1 = 0.9
        self.gamma2 = 0.2

        self.t1 = 0.4
        self.t2 = 0.6
        self.beta = 0.5

        self.relation_linear = nn.Linear(init_dim * 2, init_dim)  # 看情况调整
        self.proximity_linear = nn.Linear(init_dim * 2, init_dim)

        self.rela_gcn = TPTGCN(init_dim, self.hidden_dim, self.output_dim)

        self.device = device

        self.input_encoder = Encoder(e_dim=self.output_dim,
                                     n_layers=1, n_head=8, d_k=32, d_v=32,
                                     dropout=0)
        self.output_encoder = Encoder(e_dim=self.output_dim,
                                      n_layers=1, n_head=8, d_k=32, d_v=32,
                                      dropout=0)

    # def get_align_neg(self, sim, axis):
    #
    #     sim_rank = np.argsort(sim, axis=axis)[:, :self.neg_k]
    #     pos_temp = np.arange(sim.shape[0])
    #     pos_temp = np.repeat(pos_temp, self.neg_k)
    #
    #     neg = sim_rank.reshape((-1,))
    #     pos = pos_temp.reshape((-1,))
    #
    #     if axis == 0:
    #         return neg, pos
    #     elif axis == 1:
    #         return pos, neg

    def get_align_neg(self, source_features, target_features):

        sample_len = source_features.size()[0]

        # neg_temp = torch.tensor([], dtype=torch.long).to(self.device)
        # for i in range(sample_len):
        #     this_source_feature = source_features[i]
        #     manhattan_sim = torch.abs(this_source_feature - target_features)
        #     manhattan_sim = torch.sum(manhattan_sim, 1)
        #     sim_rank = torch.argsort(manhattan_sim)[:self.neg_k]
        #     neg_temp = torch.cat((neg_temp, sim_rank))
        cos_sim = cosine_distance(source_features, target_features)
        sim_rank = torch.argsort(cos_sim, dim=1)[:, -self.neg_k:]

        # argsort(sim, axis=axis)[:, :self.neg_k]

        pos_temp = torch.arange(sample_len).to(self.device)
        pos_temp = pos_temp.repeat(1, self.neg_k)

        neg = sim_rank.contiguous().view((-1,))
        pos = pos_temp.view((-1,))

        return pos, neg

    def get_proximity(self, input_proximity, output_proximity, entity_features, relation_features):
        global_entity_em = torch.zeros(len(input_proximity), self.output_dim * 2, device=self.device)
        for i, entity_id in enumerate(input_proximity):

            kernel_em = entity_features[entity_id]

            init_input_seq = np.array(input_proximity[entity_id])
            init_output_seq = np.array(output_proximity[entity_id])

            if len(init_input_seq) == 0:
                input_em = torch.zeros(self.output_dim, device=self.device)
            else:
                relation_input_seq = init_input_seq[:, 0]
                head_input_seq = init_input_seq[:, 1]

                relation_input_seq = torch.tensor(relation_input_seq, dtype=torch.long).to(self.device)
                head_input_seq = torch.tensor(head_input_seq, dtype=torch.long).to(self.device)

                input_em = self.input_encoder(relation_input_seq, head_input_seq, kernel_em, entity_features,
                                              relation_features)

            if len(init_output_seq) == 0:
                output_em = torch.zeros(self.output_dim, device=self.device)
            else:
                relation_output_seq = init_output_seq[:, 0]
                head_output_seq = init_output_seq[:, 1]

                relation_output_seq = torch.tensor(relation_output_seq, dtype=torch.long).to(self.device)
                head_output_seq = torch.tensor(head_output_seq, dtype=torch.long).to(self.device)

                output_em = self.output_encoder(relation_output_seq, head_output_seq, kernel_em, entity_features,
                                                relation_features)


            temp_em = torch.cat((input_em, output_em))
            proximity_em = self.proximity_linear(temp_em)
            global_entity_em[i] = torch.cat((kernel_em, proximity_em))
            # global_entity_em[i] = temp_em

        return global_entity_em

    def get_align_loss(self, entity_features, relation_features, train_proximity, train_len):
        # l_i, l_o, r_i, r_o = proximity_tup

        l_train_i, l_train_o, r_train_i, r_train_o = train_proximity
        train_left_proximity_em = self.get_proximity(l_train_i, l_train_o, entity_features, relation_features)
        train_right_proximity_em = self.get_proximity(r_train_i, r_train_o, entity_features, relation_features)

        # pos_part = torch.sum(torch.abs(train_left_proximity_em - train_right_proximity_em), 1) + self.gamma
        post_sim_matrix = cosine_distance(train_left_proximity_em, train_right_proximity_em)
        post_sim_diag = torch.einsum("ii->i", post_sim_matrix)
        pos_part = torch.mean(torch.clamp(post_sim_diag, max=self.gamma1))

        no_diag_sim_matrix = post_sim_matrix - torch.diag(post_sim_diag)
        neg_part = torch.mean(torch.clamp(no_diag_sim_matrix, min=self.gamma2))

        total_loss = neg_part - self.gamma2 - pos_part + self.gamma1
        '''
        with torch.no_grad():
            pl, nr = self.get_align_neg(train_left_proximity_em, train_right_proximity_em)
            nl, pr = self.get_align_neg(train_right_proximity_em, train_left_proximity_em)

        neg_left_features_pl = torch.index_select(train_left_proximity_em, 0, pl)
        neg_left_features_nr = torch.index_select(train_right_proximity_em, 0, nr)

        neg_left_features_nl = torch.index_select(train_left_proximity_em, 0, nl)
        neg_left_features_pr = torch.index_select(train_right_proximity_em, 0, pr)

        neg_nr_pl_sim_matrix = cosine_distance(neg_left_features_nr, neg_left_features_pl)
        neg_nr_pl_sim_diag = torch.einsum("ii->i", neg_nr_pl_sim_matrix)
        neg_nr_pl_part = torch.mean(torch.clamp(neg_nr_pl_sim_diag, min=self.gamma2))

        # neg_part_nr_pl = -torch.sum(torch.abs(neg_left_features_nr - neg_left_features_pl), 1)

        neg_nl_pr_sim_matrix = cosine_distance(neg_left_features_nl, neg_left_features_pr)
        neg_nl_pr_sim_diag = torch.einsum("ii->i", neg_nl_pr_sim_matrix)
        neg_nl_pr_part = torch.mean(torch.clamp(neg_nl_pr_sim_diag, min=self.gamma2))

        # neg_part_nl_pr = -torch.sum(torch.abs(neg_left_features_nl - neg_left_features_pr), 1)

        # loss1 = torch.relu(torch.add(pos_part.view([train_len, 1]), neg_part_nr_pl.view([train_len, self.neg_k])))
        # loss2 = torch.relu(torch.add(pos_part.view([train_len, 1]), neg_part_nl_pr.view([train_len, self.neg_k])))

        # total_loss = torch.add(torch.sum(loss1), torch.sum(loss2)) / (2.0 * train_len * self.neg_k)

        total_loss = (-2*pos_part + neg_nr_pl_part + neg_nl_pr_part)/2 + self.gamma1 - self.gamma2
        '''
        return total_loss, train_left_proximity_em, train_right_proximity_em

    def get_test_em(self, test_proximity_tup):

        l_test_i, l_test_o, r_test_i, r_test_o = test_proximity_tup
        train_left_proximity_em = self.get_proximity(l_test_i, l_test_o, self.ex, self.rx)
        train_right_proximity_em = self.get_proximity(r_test_i, r_test_o, self.ex, self.rx)

        return train_left_proximity_em, train_right_proximity_em

    def forward(self, e_x, r_x, prim_adj, rela_adj, train_len, proximity_tup):
        # gcn_ex = self.prim_gcn(e_x, prim_adj)

        init_relation_x = self.relation_linear(r_x)
        # gcn_layer_output = torch.cat((e_x, init_relation_x), 0)

        rela_gcn_x = self.rela_gcn(e_x, init_relation_x, prim_adj, rela_adj)
        gcn_ex = rela_gcn_x[:self.eitity_cnt]
        gcn_rx = rela_gcn_x[self.eitity_cnt:]

        align_loss, lem, rem = self.get_align_loss(gcn_ex, gcn_rx, proximity_tup, train_len)
        self.ex = gcn_ex
        self.rx = gcn_rx

        return align_loss, lem, rem
