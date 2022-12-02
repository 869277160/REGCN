import torch.nn.functional as F
import torch

from utils import *
from pp_JLER import *
from Config import *
from models import *
from evaluation import *

from torch import optim
from datetime import datetime
import time



'''
2020-06-08 23:36:28
这个是DASFAA的论文模型
'''
def run_tradic_gcn_model(entity_cnt, train_data, test_data, kg1, kg2):
# def run_hybrid_model03(Config, entity_cnt, train_data, test_data, merge_kg, kg1, kg2):

    language = JLER_Config.language
    learning_rate = JLER_Config.learning_rate
    epochs = JLER_Config.epochs
    device = JLER_Config.device
    dimension = JLER_Config.dim

    merge_kg =  kg1 + kg2

    train_data = torch.LongTensor(train_data).to(device)
    # test_data = np.array(test_data)
    test_data = torch.LongTensor(test_data).to(device)

    # entity_features = get_node_feature(entity_cnt, dimension, language)
    # relation_features = get_RDGCN_edge_feature(merge_kg, entity_features, entity_cnt)

    pwd = '/root/mo/REGCN/'
    entity_features = process_pickle(pwd+"output/entity_X.pkl")
    relation_features = process_pickle(pwd+"output/relation_X.pkl")

    proximity_tup_all = get_neighbor_aware_proximity_split(train_data, test_data, entity_cnt, kg1, kg2)

    '''
    adjacency of Primal Graph
    '''
    primgraph_adj = get_primal_graph_adj(entity_cnt, merge_kg)

    '''
    adjacency of Relation Graph
    '''
    # relagraph_adj = get_relation_graph_adj(entity_cnt, merge_kg)
    dualgraph_adj = get_duality_graph_adj(entity_cnt, merge_kg)

    entity_features = torch.tensor(entity_features, dtype=torch.float32).to(device)
    relation_features = torch.tensor(relation_features, dtype=torch.float32).to(device)
    primgraph_adj = primgraph_adj.to(device)
    # relagraph_adj = relagraph_adj.to(device)
    dualgraph_adj = dualgraph_adj.to(device)


    model = TradicGCN(dimension, entity_cnt, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    last_datetime = datetime.now()

    print("hybrid graph model: training...start...")
    for epoch in range(epochs):
        this_datetime = datetime.now()
        time_period = (this_datetime - last_datetime).seconds

        print("curent time:", this_datetime)
        print("time period:", time_period)

        l_train_i, l_train_o, l_test_i, l_test_o, r_train_i, r_train_o, r_test_i, r_test_o = proximity_tup_all
        model.train(True)
        # l_train_i_batch, l_train_o_batch, r_train_i_batch, r_train_o_batch = get_train_proximity_batch(l_train_i, l_train_o, r_train_i, r_train_o)
        # train_proximity_tup = get_train_proximity_batch(l_train_i, l_train_o, r_train_i, r_train_o, train_data, batch_size=500)
        l_train_i, l_train_o, r_train_i, r_train_o = get_train_proximity_batch(l_train_i, l_train_o, r_train_i,
                                                                               r_train_o, train_data,
                                                                               batch_size=500)

        print(epoch, list(l_train_i[0].keys())[0])

        for batch in range(len(l_train_i)):
            batch_proximity_tup = [l_train_i[batch], l_train_o[batch], r_train_i[batch], r_train_o[batch]]
            optimizer.zero_grad()
            train_len = len(l_train_i[batch])
            # loss, train_lem, train_rem = model(entity_features, relation_features, primgraph_adj, relagraph_adj, train_len, batch_proximity_tup)
            loss, train_lem, train_rem = model(entity_features, relation_features, primgraph_adj, dualgraph_adj,
                                               train_len, batch_proximity_tup)

            # print(epoch, loss.item())
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.train(False)
            with torch.no_grad():
                test_proximity_tup = [l_test_i, l_test_o, r_test_i, r_test_o]
                # test_lem, test_rem = model.get_test_em(entity_features, relation_features, primgraph_adj, relagraph_adj, test_data, test_proximity_tup)
                test_lem, test_rem = model.get_test_em(test_proximity_tup)
                #
                #     cos_matrix = cosine_distance(test_lem, test_rem)
                #     accuracy_m, correct_num, num_user = evalation(cos_matrix, top_k=1, device=device)
                #     print('accuracy: %.3f' % accuracy_m)
                #     print('correct_num: %d' % correct_num)
                #     print('num_user: %d' % num_user)
                test_lem_np = test_lem.cpu().detach().numpy()
                test_rem_np = test_rem.cpu().detach().numpy()
                test_data_np = test_data.cpu().detach().numpy()

            # outvec_numpy = outvec.cpu().detach().numpy()
            # get_hits3(test_lem_np, test_rem_np, test_data_np)
            # cos_get_hits(test_lem_np, test_rem_np, test_data_np)
            cos_get_hits_mrr(test_lem_np, test_rem_np, test_data_np)

        print('%d/%d' % (epoch + 1, epochs), 'epochs...', loss.item())


if __name__ == '__main__':

    '''
    JLER 数据集的数据预处理，用来构建训练集和测试集
    '''
    need_relation = True

 
    source_KG, target_KG, train_aligns, test_aligns, r_train_aligns, r_test_aligns, r_aligns, info_dict, entity_cnt = JLER_train_test_set_construction(need_relation)

    '''
    模型入口：
    run_tradic_gcn_model：使用 三元图作为图神经网络处理结构
        数据集：JLER
    '''
    run_tradic_gcn_model(entity_cnt, train_aligns, test_aligns, source_KG, target_KG)