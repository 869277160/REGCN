import numpy as np
import json

import torch.nn.functional as F
import torch
import torch.nn as nn

from IOUtils import *
from sklearn import preprocessing

import scipy.sparse as sp
import scipy

# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret

def load_entity(fn, num=1):
    print('loading entities from a file...' + fn)
    entities = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            for i in range(num):
                entities.append(int(th[i]))
    return entities

def get_input_layer(e, dimension, lang):
    print('adding the primal input layer...')
    with open(file='../data/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    # input_embeddings = tf.convert_to_tensor(embedding_list)
    # ent_embeddings = tf.Variable(input_embeddings)

    input_embeddings = torch.tensor(embedding_list, dtype=torch.float32)

    return F.normalize(input_embeddings, p=2, dim=1)

# used:2020-3-18 18:19:52
# def get_node_feature(e, dimension, language):
def get_node_feature(language):
    lang = language[:2]
    with open(file='./data/JLER/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')

    normalized_embedding = preprocessing.normalize(np.array(embedding_list), norm='l2', axis=1)
    generate_pickle("output/entity_X.pkl", normalized_embedding)
    return normalized_embedding

# used:2020-3-18 18:19:52
def relation_func(KG, e):
    head = {}
    tail = {}
    relation_cnt_dict = {}

    for tri in KG:  # 得到tri的relation的 head节点 和 tail节点，并relation出现次数计数
        if tri[1] not in relation_cnt_dict:
            relation_cnt_dict[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            relation_cnt_dict[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])
    # r_num = len(head) # 有3024个不同的边
    # head_r = np.zeros((e, r_num))
    # tail_r = np.zeros((e, r_num))
    # r_mat_ind = []
    # r_mat_val = []
    # for tri in KG: # 构造head 和relation的关系矩阵，构造tail和...
    #     head_r[tri[0]][tri[1]] = 1
    #     tail_r[tri[2]][tri[1]] = 1
    #     r_mat_ind.append([tri[0], tri[2]])
    #     r_mat_val.append(tri[1])
    # # r_mat = tf.SparseTensor(
    # #     indices=r_mat_ind, values=r_mat_val, dense_shape=[e, e])
    #
    # ind_LongTensor = torch.LongTensor(np.array(r_mat_ind))
    # val_FloatTensor = torch.FloatTensor(np.array(r_mat_val))
    # r_mat = torch.sparse.FloatTensor(ind_LongTensor.t(), val_FloatTensor, torch.Size([e, e]))

    # return head, tail, head_r, tail_r, r_mat

    relation_cnt = len(relation_cnt_dict)

    # print(np.max(list(relation_cnt_dict.keys())))
    # print(relation_cnt)

    return relation_cnt

def get_edge_feature_pyt(triples, node_features, entity_cnt):

    # head, tail, head_r, tail_r, r_mat = rfunc(triples, entity_cnt)
    # dual_X, dual_A = get_dual_input(
    #     primal_X_0, head, tail, head_r, tail_r, dimension)

    relation_cnt = relation_func(triples, entity_cnt)
    relation_X_dict = {}

    relation_X = torch.zeros(relation_cnt, node_features.size()[1])

    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if relation not in relation_X_dict:
            relation_X_dict[relation]["head"] = node_features[head]
            relation_X_dict[relation]["tail"] = node_features[tail]
        else:
            relation_X_dict[relation]["head"] = torch.cat((relation_X_dict[relation]["head"], node_features[head]))
            relation_X_dict[relation]["tail"] = torch.cat((relation_X_dict[relation]["tail"], node_features[tail]))

    for i, relation in enumerate(relation_X_dict):

        feature_tup = relation_X_dict[relation]

        head_feature = feature_tup["head"].view(len(feature_tup["head"]), -1)
        tail_feature = feature_tup["tail"].view(len(feature_tup["tail"]), -1)

        head_feature = torch.mean(head_feature, 0)
        tail_feature = torch.mean(tail_feature, 0)

        relation_X[relation] = torch.cat((head_feature, tail_feature))

    return relation_X

def get_edge_feature(triples, node_features, entity_cnt):
    relation_cnt = relation_func(triples, entity_cnt)
    relation_X_dict = {}

    # relation_X = np.zeros((relation_cnt, node_features.shape[1]*2))
    relation_X = np.zeros((relation_cnt, node_features.shape[1]))

    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if relation not in relation_X_dict:
            relation_X_dict[relation] = {}
            relation_X_dict[relation]["head"] = node_features[head]
            relation_X_dict[relation]["tail"] = node_features[tail]
        else:
            # a = relation_X_dict[relation]["head"]
            # b = node_features[head]
            # c = np.concatenate((relation_X_dict[relation]["head"], node_features[head]))
            relation_X_dict[relation]["head"] = np.concatenate((relation_X_dict[relation]["head"], node_features[head]))
            relation_X_dict[relation]["tail"] = np.concatenate((relation_X_dict[relation]["tail"], node_features[tail]))

    for i, relation in enumerate(relation_X_dict):
        feature_tup = relation_X_dict[relation]

        # head_feature = feature_tup["head"].view(len(feature_tup["head"]), -1)
        # head_feature = np.reshape(feature_tup["head"], (len(feature_tup["head"]), -1))
        head_feature = np.reshape(feature_tup["head"], (-1, node_features.shape[1]))

        # tail_feature = feature_tup["tail"].view(len(feature_tup["tail"]), -1)
        # tail_feature = np.reshape(feature_tup["tail"], (len(feature_tup["tail"]), -1))
        tail_feature = np.reshape(feature_tup["tail"], (-1, node_features.shape[1]))

        '''
        2019-10-24 15:30:07
        论文里的relation features方式：
        '''
        '''
        # head_feature = torch.mean(head_feature, 0)
        head_feature = np.mean(head_feature, axis=0)
        # tail_feature = torch.mean(tail_feature, 0)
        tail_feature = np.mean(tail_feature, axis=0)

        # relation_X[relation] = torch.cat((head_feature, tail_feature))
        relation_X[relation] = np.concatenate((head_feature, tail_feature))
        '''

        '''
        2019-10-24 15:30:56
        用相减的方式来构建
        '''
        relation_feature = tail_feature - head_feature
        relation_X[relation] = np.mean(relation_feature, axis=0)


    return relation_X

# used:2020-3-18 18:19:52
def get_RDGCN_edge_feature(triples, node_features, entity_cnt):
    relation_cnt = relation_func(triples, entity_cnt)
    relation_X_dict = {}

    relation_X = np.zeros((relation_cnt, node_features.shape[1]*2))
    # relation_X = np.zeros((relation_cnt, node_features.shape[1]))

    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if relation not in relation_X_dict:
            relation_X_dict[relation] = {}
            relation_X_dict[relation]["head"] = node_features[head]
            relation_X_dict[relation]["tail"] = node_features[tail]
        else:
            # a = relation_X_dict[relation]["head"]
            # b = node_features[head]
            # c = np.concatenate((relation_X_dict[relation]["head"], node_features[head]))
            relation_X_dict[relation]["head"] = np.concatenate((relation_X_dict[relation]["head"], node_features[head]))
            relation_X_dict[relation]["tail"] = np.concatenate((relation_X_dict[relation]["tail"], node_features[tail]))

    for i, relation in enumerate(relation_X_dict):
        feature_tup = relation_X_dict[relation]

        # head_feature = feature_tup["head"].view(len(feature_tup["head"]), -1)
        # head_feature = np.reshape(feature_tup["head"], (len(feature_tup["head"]), -1))
        head_feature = np.reshape(feature_tup["head"], (-1, node_features.shape[1]))

        # tail_feature = feature_tup["tail"].view(len(feature_tup["tail"]), -1)
        # tail_feature = np.reshape(feature_tup["tail"], (len(feature_tup["tail"]), -1))
        tail_feature = np.reshape(feature_tup["tail"], (-1, node_features.shape[1]))

        '''
        2019-10-24 15:30:07
        论文里的relation features方式：
        '''
        # head_feature = torch.mean(head_feature, 0)
        head_feature = np.mean(head_feature, axis=0)
        # tail_feature = torch.mean(tail_feature, 0)
        tail_feature = np.mean(tail_feature, axis=0)

        # relation_X[relation] = torch.cat((head_feature, tail_feature))
        relation_X[relation] = np.concatenate((head_feature, tail_feature))

        '''
        2019-10-24 15:30:56
        用相减的方式来构建
        '''
        # relation_feature = tail_feature - head_feature
        # relation_X[relation] = np.mean(relation_feature, axis=0)

    # generate_pickle("output/relation_X.pkl", relation_X)
    return relation_X

def get_heter_relation_features(relation_features, relation_dict):
    heter_relation_features = []
    for heter_rel_id, triple_tup in relation_dict.items():
        relation_id = triple_tup[1]
        this_relation_features = relation_features[relation_id]
        heter_relation_features.append(this_relation_features)

    heter_relation_features = np.array(heter_relation_features)

    return heter_relation_features


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_relation_graph_adj(entity_cnt, triples):
    edges_tup = []
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        relation_id = entity_cnt + relation
        edges_tup.append([head, relation_id])
        edges_tup.append([tail, relation_id])

    relation_cnt = relation_func(triples, entity_cnt)

    # edges_unordered = np.ndarray(edges_tup)
    edges_unordered = np.array(edges_tup)
    edges = edges_unordered

    adj_dim = entity_cnt + relation_cnt
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(adj_dim, adj_dim),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

'''
2020-3-20 12:09:22
用来生成异构关系的adj
'''
def get_heterogeneous_relation_graph_adj(entity_cnt, triples):
    edges_tup = []
    relation_dict = {}
    relation_dict_reverse = {}
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        heter_rel_id = len(relation_dict)

        '''
        2020-3-28 13:50:34
        临时测试
        '''
        # if head > 30000 or tail > 30000:
        #     print(triple)

        relation_dict[heter_rel_id] = [head, relation, tail]
        if relation not in relation_dict_reverse:
            relation_dict_reverse[relation] = []
        relation_dict_reverse[relation].append(heter_rel_id)

        '''
        2020-3-20 18:11:38
        我发现一个严重的错误，这里的relation_id不应该用entity_cnt + relation
        因为，实际上entity的最大id并不等于entity_cnt
        
        2020-3-27 18:08:33
        entity cnt是triples中entity的数量，有很多entity实际上并没有参与到三元组中
        '''
        entity_id_plus = 38960

        relation_id = entity_id_plus + heter_rel_id
        edges_tup.append([head, relation_id])
        edges_tup.append([tail, relation_id])

    relation_cnt = len(relation_dict)

    edges_unordered = np.array(edges_tup)
    edges = edges_unordered

    adj_dim = entity_cnt + relation_cnt
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(adj_dim, adj_dim),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, relation_dict, relation_dict_reverse

def get_primal_graph_adj(entity_cnt, triples):

    edges_tup = []
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        relation_id = entity_cnt + relation
        # edges_tup.append([head, relation_id])
        # edges_tup.append([tail, relation_id])
        edges_tup.append([head, tail])

    # edges_unordered = np.ndarray(edges_tup)
    edges_unordered = np.array(edges_tup)
    edges = edges_unordered

    adj_dim = entity_cnt
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(adj_dim, adj_dim),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # generate_pickle("output/adj.pkl", adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def GAT_load_data(entity_cnt, triples):
    # GAT load data process:
    edges_tup = []
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        relation_id = entity_cnt + relation
        # edges_tup.append([head, relation_id])
        # edges_tup.append([tail, relation_id])
        edges_tup.append([head, tail])

    # edges_unordered = np.ndarray(edges_tup)
    edges_unordered = np.array(edges_tup)
    edges = edges_unordered

    adj_dim = entity_cnt
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(adj_dim, adj_dim),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    return  adj.cuda()

def get_bi_context(entity_ids, kg):

    input_context = {}
    input_context_unordered = {}
    output_context = {}
    output_context_unordered = {}

    for triple in kg:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if head not in output_context_unordered:
            output_context_unordered[head] = [[relation, tail]]
        else:
            output_context_unordered[head].append([relation, tail])

        if tail not in input_context_unordered:
            input_context_unordered[tail] = [[relation, head]]
        else:
            input_context_unordered[tail].append([relation, head])

    entity_ids_np = entity_ids.cpu().detach().numpy()

    for entity_id in entity_ids_np:

        if entity_id not in input_context_unordered:
            input_context[entity_id] = []
        else:
            input_context[entity_id] = input_context_unordered[entity_id]

        if entity_id not in output_context_unordered:
            output_context[entity_id] = []
        else:
            output_context[entity_id] = output_context_unordered[entity_id]

    return input_context, output_context

def get_context_split(entity_set1, entity_set2, kg1, kg2):

    input_context1, output_context1 = get_bi_context(entity_set1, kg1)
    input_context2, output_context2 = get_bi_context(entity_set2, kg2)

    return input_context1, output_context1, input_context2, output_context2

def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def manhattan_distance(x1, x2, eps=1e-8):
    # x2 = x1 if x2 is None else x2
    # w1 = x1.norm(p=2, dim=1, keepdim=True)
    # w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)

    # TODO: 实现 nxd & mxd => nxm 的distance计算

    # return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    return []

'''
2020-5-27 11:33:57

for mo_GAT
'''
def get_entity_nighbors(entity_cnt, triples):

    entity_input_proximity = {}
    entity_output_proximity = {}
    edges_tup = []
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        relation_id = entity_cnt + relation
        # edges_tup.append([head, relation_id])
        # edges_tup.append([tail, relation_id])
        edges_tup.append([head, tail])

        if head not in entity_output_proximity:
            entity_output_proximity[head] = []
        else:
            entity_output_proximity[head].append(tail)

        if tail not in entity_input_proximity:
            entity_input_proximity[tail] = []
        else:
            entity_input_proximity[tail].append(head)



    # edges_unordered = np.ndarray(edges_tup)
    edges_unordered = np.array(edges_tup)
    edges = edges_unordered

    adj_dim = entity_cnt
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(adj_dim, adj_dim),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # generate_pickle("output/adj.pkl", adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # 将dict 清洗成数组
    entity_proximity = []
    for entity_index in range(entity_cnt):

        if entity_index not in entity_input_proximity and entity_index not in entity_output_proximity:
            print("bug:2020-5-27 12:07:58")

        try:
            input_proximity = entity_input_proximity[entity_index]
        except Exception:
            input_proximity = []

        try:
            output_proximity = entity_output_proximity[entity_index]
        except Exception:
            output_proximity = []

        # 如果是无序的，代码这样写
        this_entity_proximity = list(set(input_proximity + output_proximity))
        entity_proximity.append(this_entity_proximity)

    return adj, entity_proximity

'''
2020-5-28 11:21:43

for RGAT
'''

def get_proximity(entity_cnt, triples):
    entity_input_proximity = {}
    entity_output_proximity = {}
    relation_proximity = {}
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        relation_id = entity_cnt + relation
        if head not in entity_output_proximity:
            entity_output_proximity[head] = []
            entity_output_proximity[head].append([relation_id, tail])
        else:
            entity_output_proximity[head].append([relation_id, tail])

        if tail not in entity_input_proximity:
            entity_input_proximity[tail] = []
            entity_input_proximity[tail].append([relation_id, head])
        else:
            entity_input_proximity[tail].append([relation_id, head])

        if relation not in relation_proximity:
            relation_proximity[relation] = []
            relation_proximity[relation].append([head, tail])
        else:
            relation_proximity[relation].append([head, tail])




    # 将dict 清洗成数组
    ei_proximity = {}
    eo_proximity = {}
    for entity_index in range(entity_cnt):
        try:
            ei_proximity[entity_index] = np.array(entity_input_proximity[entity_index])
        except Exception:
            ei_proximity[entity_index] = np.array([])

        try:
            eo_proximity[entity_index] = np.array(entity_output_proximity[entity_index])
        except Exception:
            eo_proximity[entity_index] = np.array([])

    r_proximity = {}
    for relation_index in range(len(relation_proximity)):
        r_proximity[entity_cnt+relation_index] = np.array(relation_proximity[relation_index])

        # 如果是无序的，代码这样写
        # this_entity_proximity = list(set(input_proximity + output_proximity))
        # entity_proximity.append(this_entity_proximity)

    return ei_proximity, eo_proximity, r_proximity

def get_train_batch(train_data, batch_size=500):
    # train_ids_np = train_data.cpu().detach().numpy()
    np.random.shuffle(train_data)
    this_batch_i = -1
    train_data_batch = {}
    for i, train_id_tup in enumerate(train_data):

        # if i in [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
        if i in list(range(0, len(train_data), batch_size)):
            this_batch_i += 1
            train_data_batch[this_batch_i] = []
        train_data_batch[this_batch_i].append(train_id_tup)

    return train_data_batch

def get_test(test_data, ex):
    test_left = test_data[:, 0]
    test_right = test_data[:, 1]

    test_left_em = ex[test_left]
    test_right_em = ex[test_right]

    return test_left_em, test_right_em


'''
2020-06-08 23:42:32

for tradicGCN
'''
def get_kg_proximity(kg, entity_ids):

    input_proximity = {}
    output_proximity = {}

    for triple in kg:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if head in entity_ids:
            if head not in output_proximity:
                output_proximity[head] = [[relation, tail]]
            else:
                output_proximity[head].append([relation, tail])

        if tail in entity_ids:
            if tail not in input_proximity:
                input_proximity[tail] = [[relation, head]]
            else:
                input_proximity[tail].append([relation, head])

    for entity_id in entity_ids:
        if entity_id not in input_proximity:
            input_proximity[entity_id] = []
        if entity_id not in output_proximity:
            output_proximity[entity_id] = []

    return input_proximity, output_proximity

def get_kg_proximity_split(kg, train_ids, test_ids):

    train_input_proximity_unordered = {}
    train_output_proximity_unordered = {}

    test_input_proximity_unordered = {}
    test_output_proximity_unordered = {}

    train_input_proximity = {}
    train_output_proximity = {}

    test_input_proximity = {}
    test_output_proximity = {}


    for triple in kg:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if head in train_ids:
            if head not in train_output_proximity_unordered:
                train_output_proximity_unordered[head] = [[relation, tail]]
            else:
                train_output_proximity_unordered[head].append([relation, tail])
        elif head in test_ids:
            if head not in test_output_proximity_unordered:
                test_output_proximity_unordered[head] = [[relation, tail]]
            else:
                test_output_proximity_unordered[head].append([relation, tail])

        if tail in train_ids:
            if tail not in train_input_proximity_unordered:
                train_input_proximity_unordered[tail] = [[relation, head]]
            else:
                train_input_proximity_unordered[tail].append([relation, head])
        elif tail in test_ids:
            if tail not in test_input_proximity_unordered:
                test_input_proximity_unordered[tail] = [[relation, head]]
            else:
                test_input_proximity_unordered[tail].append([relation, head])

    train_ids_np = train_ids.cpu().detach().numpy()
    test_ids_np = test_ids.cpu().detach().numpy()

    for train_id in train_ids_np:

        if train_id not in train_input_proximity_unordered:
            train_input_proximity[train_id] = []
        else:
            train_input_proximity[train_id] = train_input_proximity_unordered[train_id]

        if train_id not in train_output_proximity_unordered:
            train_output_proximity[train_id] = []
        else:
            train_output_proximity[train_id] = train_output_proximity_unordered[train_id]

    for test_id in test_ids_np:
        if test_id not in test_input_proximity_unordered:
            test_input_proximity[test_id] = []
        else:
            test_input_proximity[test_id] = test_input_proximity_unordered[test_id]
        if test_id not in test_output_proximity_unordered:
            test_output_proximity[test_id] = []
        else:
            test_output_proximity[test_id] = test_output_proximity_unordered[test_id]


    return train_input_proximity, train_output_proximity, test_input_proximity, test_output_proximity

def get_kg_proximity_split_np(kg, train_ids, test_ids):

    train_input_proximity_unordered = {}
    train_output_proximity_unordered = {}

    test_input_proximity_unordered = {}
    test_output_proximity_unordered = {}

    train_input_proximity = {}
    train_output_proximity = {}

    test_input_proximity = {}
    test_output_proximity = {}

    for triple in kg:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if head in train_ids:
            if head not in train_output_proximity_unordered:
                train_output_proximity_unordered[head] = [[relation, tail]]
            else:
                train_output_proximity_unordered[head].append([relation, tail])
        elif head in test_ids:
            if head not in test_output_proximity_unordered:
                test_output_proximity_unordered[head] = [[relation, tail]]
            else:
                test_output_proximity_unordered[head].append([relation, tail])

        if tail in train_ids:
            if tail not in train_input_proximity_unordered:
                train_input_proximity_unordered[tail] = [[relation, head]]
            else:
                train_input_proximity_unordered[tail].append([relation, head])
        elif tail in test_ids:
            if tail not in test_input_proximity_unordered:
                test_input_proximity_unordered[tail] = [[relation, head]]
            else:
                test_input_proximity_unordered[tail].append([relation, head])

    train_ids_np = train_ids
    test_ids_np = test_ids

    for train_id in train_ids_np:

        if train_id not in train_input_proximity_unordered:
            train_input_proximity[train_id] = []
        else:
            train_input_proximity[train_id] = train_input_proximity_unordered[train_id]

        if train_id not in train_output_proximity_unordered:
            train_output_proximity[train_id] = []
        else:
            train_output_proximity[train_id] = train_output_proximity_unordered[train_id]

    for test_id in test_ids_np:
        if test_id not in test_input_proximity_unordered:
            test_input_proximity[test_id] = []
        else:
            test_input_proximity[test_id] = test_input_proximity_unordered[test_id]
        if test_id not in test_output_proximity_unordered:
            test_output_proximity[test_id] = []
        else:
            test_output_proximity[test_id] = test_output_proximity_unordered[test_id]


    return train_input_proximity, train_output_proximity, test_input_proximity, test_output_proximity

def get_neighbor_aware_proximity(total_eneities, kg1, kg2):

    left_ids = total_eneities[:, 0]
    right_ids = total_eneities[:, 1]

    left_input_proximity, left_output_proximity = get_kg_proximity(kg1, left_ids)
    right_input_proximity, right_output_proximity = get_kg_proximity(kg2, right_ids)

    return left_input_proximity, left_output_proximity, right_input_proximity, right_output_proximity

def get_neighbor_aware_proximity_split(train_data, test_data, entity_cnt, kg1, kg2):

    left_train_ids = train_data[:, 0]
    right_train_ids = train_data[:, 1]
    left_test_ids = test_data[:, 0]
    right_test_ids = test_data[:, 1]

    left_train_input_proximity, left_train_output_proximity, left_test_input_proximity, left_test_output_proximity = \
        get_kg_proximity_split(kg1, left_train_ids, left_test_ids)

    right_train_input_proximity, right_train_output_proximity, right_test_input_proximity, right_test_output_proximity = \
        get_kg_proximity_split(kg2, right_train_ids, right_test_ids)

    return left_train_input_proximity, left_train_output_proximity, left_test_input_proximity, \
           left_test_output_proximity, right_train_input_proximity, right_train_output_proximity, \
           right_test_input_proximity, right_test_output_proximity

def get_neighbor_aware_proximity_split_np(train_data, test_data, entity_cnt, kg1, kg2):

    left_train_ids = train_data[:, 0]
    right_train_ids = train_data[:, 1]
    left_test_ids = test_data[:, 0]
    right_test_ids = test_data[:, 1]

    left_train_input_proximity, left_train_output_proximity, left_test_input_proximity, left_test_output_proximity = \
        get_kg_proximity_split_np(kg1, left_train_ids, left_test_ids)

    right_train_input_proximity, right_train_output_proximity, right_test_input_proximity, right_test_output_proximity = \
        get_kg_proximity_split_np(kg2, right_train_ids, right_test_ids)

    return left_train_input_proximity, left_train_output_proximity, left_test_input_proximity, \
           left_test_output_proximity, right_train_input_proximity, right_train_output_proximity, \
           right_test_input_proximity, right_test_output_proximity


def get_duality_graph_adj(entity_cnt, triples):
    edges_tup = []
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        '''
        2020-3-28 13:50:34
        临时测试
        '''
        # if head > 30000 or tail > 30000:
        #     print(triple)

        relation_id = entity_cnt + relation
        edges_tup.append([head, relation_id])
        edges_tup.append([tail, relation_id])
        edges_tup.append([head, tail])

    relation_cnt = relation_func(triples, entity_cnt)

    # edges_unordered = np.ndarray(edges_tup)
    edges_unordered = np.array(edges_tup)
    edges = edges_unordered

    adj_dim = entity_cnt + relation_cnt
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(adj_dim, adj_dim),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def get_lr_train_batch(train_i, train_o, train_ids, batch_size):

    this_batch_i = -1

    train_input_proximity = {}
    train_output_proximity = {}

    for i, train_id in enumerate(train_ids):

        # if i in [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
        if i in list(range(0, len(train_ids), batch_size)):
            this_batch_i += 1
            train_input_proximity[this_batch_i] = {}
            train_output_proximity[this_batch_i] = {}

        if train_id not in train_i:
            train_input_proximity[this_batch_i][train_id] = []
        else:
            train_input_proximity[this_batch_i][train_id] = train_i[train_id]

        if train_id not in train_o:
            train_output_proximity[this_batch_i][train_id] = []
        else:
            train_output_proximity[this_batch_i][train_id] = train_o[train_id]


    return train_input_proximity, train_output_proximity

def get_train_proximity_batch(l_train_i, l_train_o, r_train_i, r_train_o, train_data, batch_size):
    train_ids_np = train_data.cpu().detach().numpy()
    np.random.shuffle(train_ids_np)
    left_train_ids = train_ids_np[:, 0]
    right_train_ids = train_ids_np[:, 1]

    l_train_i_batch, l_train_o_batch = get_lr_train_batch(l_train_i, l_train_o,left_train_ids, batch_size)
    r_train_i_batch, r_train_o_batch = get_lr_train_batch( r_train_i, r_train_o, right_train_ids, batch_size)

    return l_train_i_batch, l_train_o_batch, r_train_i_batch, r_train_o_batch

'''
2020-06-11 11:21:56
for triadic hr gcn model
'''
# TODO:
#   => 清洗出heterogeneous relation triadic graph

def get_triadic_hr_graph_adj(entity_cnt, merge_kg):
    heter_relation_cnt = 165556
    relation_cnt = 3024
    edges_tup = []
    relation_dict = {}
    # relation_dict_reverse = {}
    relation_classification = np.zeros([heter_relation_cnt, relation_cnt])
    for triple in merge_kg:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        heter_rel_id = len(relation_dict)

        relation_classification[heter_rel_id, relation] = 1

        relation_dict[heter_rel_id] = [head, relation, tail]
        # if relation not in relation_dict_reverse:
        #     relation_dict_reverse[relation] = []
        # relation_dict_reverse[relation].append(heter_rel_id)

        entity_id_plus = 38960
        relation_id = entity_id_plus + heter_rel_id
        edges_tup.append([head, relation_id])
        edges_tup.append([relation_id, tail])
        edges_tup.append([head, tail])

    # relation_cnt = len(relation_dict) #165556

    edges_unordered = np.array(edges_tup)
    edges = edges_unordered

    adj_dim = entity_cnt + heter_relation_cnt
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(adj_dim, adj_dim),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, relation_dict, relation_classification

'''
2020-06-15 07:13:12

Relation Matching Consensus Model for KGA
的 数据清洗
'''
def get_relation_graph_edges(entity_cnt, triples):
    edges_tup = []
    self_rel_dict = {}
    self_rel_set = {}
    relation_dict = {}

    # entity_set = {}

    for triple_index, triple in enumerate(triples):
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        # if head not in entity_set:
        #     entity_set[head] = len(entity_set)
        # if tail not in entity_set:
        #     entity_set[tail] = len(entity_set)

        # if head not in self_rel_set:
        #     self_rel_id = head + relation
        #     edges_tup.append([head, head, self_rel_id])
        #     self_rel_set[head] = len(self_rel_set)
        #     relation_dict[self_rel_id] = []
        #     relation_dict[self_rel_id].append(len(edges_tup))
        # if tail not in self_rel_set:
        #     self_rel_id = tail + relation
        #     edges_tup.append([tail, tail, self_rel_id])
        #     self_rel_set[tail] = len(self_rel_set)
        #     relation_dict[self_rel_id] = []
        #     relation_dict[self_rel_id].append(len(edges_tup))

        edges_tup.append([head, tail, relation])

        if relation not in relation_dict:
            relation_dict[relation] = []
            relation_dict[relation].append(len(edges_tup))
        else:
            relation_dict[relation].append(len(edges_tup))


        if head not in self_rel_dict:
            self_rel_dict[head] = [[],[]]
            self_rel_dict[head][1].append(relation)
        else:
            self_rel_dict[head][1].append(relation)

        if tail not in self_rel_dict:
            self_rel_dict[tail] = [[],[]]
            self_rel_dict[tail][0].append(relation)
        else:
            self_rel_dict[tail][0].append(relation)

    relation_cnt = len(relation_dict)
    for entity_index in range(entity_cnt):
        self_rel_id = entity_index + relation_cnt
        edges_tup.append([entity_index, entity_index, self_rel_id])
        # self_rel_set[entity_index] = len(self_rel_set)
        relation_dict[self_rel_id] = [len(edges_tup)]
    # a = entity_set
    # b = entity_cnt

    # print(np.max(np.array(list(entity_set.keys()))))
    # print(np.min(np.array(list(entity_set.keys()))))

    return edges_tup, self_rel_dict, relation_dict

def get_rmcm_edge_feature(triples, node_features, entity_cnt, self_rel_dict):
    relation_cnt = relation_func(triples, entity_cnt)
    relation_X_dict = {}

    relation_X = np.zeros((relation_cnt+entity_cnt, node_features.shape[1]*2))
    # relation_X = np.zeros((relation_cnt, node_features.shape[1]))

    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if relation not in relation_X_dict:
            relation_X_dict[relation] = {}
            relation_X_dict[relation]["head"] = node_features[head]
            relation_X_dict[relation]["tail"] = node_features[tail]
        else:
            # a = relation_X_dict[relation]["head"]
            # b = node_features[head]
            # c = np.concatenate((relation_X_dict[relation]["head"], node_features[head]))
            relation_X_dict[relation]["head"] = np.concatenate((relation_X_dict[relation]["head"], node_features[head]))
            relation_X_dict[relation]["tail"] = np.concatenate((relation_X_dict[relation]["tail"], node_features[tail]))

    for i, relation in enumerate(relation_X_dict):
        feature_tup = relation_X_dict[relation]

        # head_feature = feature_tup["head"].view(len(feature_tup["head"]), -1)
        # head_feature = np.reshape(feature_tup["head"], (len(feature_tup["head"]), -1))
        head_feature = np.reshape(feature_tup["head"], (-1, node_features.shape[1]))

        # tail_feature = feature_tup["tail"].view(len(feature_tup["tail"]), -1)
        # tail_feature = np.reshape(feature_tup["tail"], (len(feature_tup["tail"]), -1))
        tail_feature = np.reshape(feature_tup["tail"], (-1, node_features.shape[1]))

        '''
        2019-10-24 15:30:07
        论文里的relation features方式：
        '''
        # head_feature = torch.mean(head_feature, 0)
        head_feature = np.mean(head_feature, axis=0)
        # tail_feature = torch.mean(tail_feature, 0)
        tail_feature = np.mean(tail_feature, axis=0)

        # relation_X[relation] = torch.cat((head_feature, tail_feature))
        relation_X[relation] = np.concatenate((head_feature, tail_feature))

        '''
        2019-10-24 15:30:56
        用相减的方式来构建
        '''
        # relation_feature = tail_feature - head_feature
        # relation_X[relation] = np.mean(relation_feature, axis=0)

    for self_rel_id, self_rel_tup in self_rel_dict.items():

        self_head_rel_tup = self_rel_tup[0]
        self_tail_rel_tup = self_rel_tup[1]

        for head_rel_id in self_head_rel_tup:
            this_head_rel_fea = relation_X[head_rel_id]
            try:
                head_rel_feature = np.concatenate((head_rel_feature, this_head_rel_fea))
            except Exception:
                head_rel_feature = this_head_rel_fea

        for tail_rel_id in self_tail_rel_tup:
            this_tail_rel_fea = relation_X[tail_rel_id]
            try:
                tail_rel_feature = np.concatenate((tail_rel_feature, this_tail_rel_fea))
            except Exception:
                tail_rel_feature = this_tail_rel_fea



        self_head_rel_feature = np.reshape(head_rel_feature, (-1, node_features.shape[1]*2))
        self_head_rel_feature = np.mean(self_head_rel_feature, axis=0)
        self_tail_rel_feature = np.reshape(tail_rel_feature, (-1, node_features.shape[1]*2))
        self_tail_rel_feature = np.mean(self_tail_rel_feature, axis=0)

        # relation_X[self_rel_id] = np.concatenate((self_head_rel_feature, self_tail_rel_feature))
        relation_X[self_rel_id] = self_head_rel_feature + self_tail_rel_feature



    generate_pickle("output/rmcm_relation_X.pkl", relation_X)
    return relation_X


'''
2020-06-08 23:40:58
×××测试代码×××
'''

def cos_get_hits_mrr(Lvec, Rvec, test_pair, top_k=(1, 10, 50, 100)):
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    top_lr = [0] * len(top_k)
    left_mrr_rank = []
    right_mrr_rank = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]

        left_mrr_rank.append(1 / (rank_index + 1))

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]

        right_mrr_rank.append(1 / (rank_index + 1))

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f%%' % (np.mean(left_mrr_rank)))

    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.4f%%' % (np.mean(right_mrr_rank)))

# def np_

def nc_cos_get_hits_mrr(Lvec, Rvec,  gcn_ex, gcn_rx, test_proximity_tup, test_pair, top_k=(1, 10, 50, 100)):
    rcsim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')

    rcsim = standardization(rcsim)
    sim_k = 20

    l_test_i, l_test_o, r_test_i, r_test_o = test_proximity_tup

    l_ent_ids = list(l_test_i.keys())
    r_ent_ids = list(r_test_i.keys())

    top_lr = [0] * len(top_k)
    left_mrr_rank = []

    for i, entity_id in enumerate(l_test_i):
        rank = rcsim[i, :].argsort()
        top_rank = rank[0: sim_k]
        # sim_rank_dict[i] = top_rank

        source_input_seq = np.array(l_test_i[entity_id])
        source_output_seq = np.array(l_test_o[entity_id])

        align_candidate = np.zeros([len(top_rank)])

        if len(source_input_seq) != 0:
            l_relation_input_seq = source_input_seq[:, 0]
            l_entity_input_seq = source_input_seq[:, 1]

            l_rel_i_em = gcn_rx[l_relation_input_seq]
            l_ent_i_em = gcn_ex[l_entity_input_seq]

            for rank_i, sim_aligns in enumerate(top_rank):
                this_ent_id = r_ent_ids[sim_aligns]
                target_input_seq = np.array(r_test_i[this_ent_id])
                # target_output_seq = np.array(r_test_o[this_ent_id])
                if len(target_input_seq) == 0:
                    align_candidate[rank_i] = 2
                    continue

                r_relation_input_seq = target_input_seq[:, 0]
                r_entity_input_seq = target_input_seq[:, 1]
                r_rel_o_em = gcn_rx[r_relation_input_seq]
                # r_ent_em = gcn_x[r_entity_input_seq]

                # rel_sim_rank = cosine_distance(l_rel_i_em, r_rel_o_em)
                rel_sim_rank = scipy.spatial.distance.cdist(l_rel_i_em, r_rel_o_em, metric='cosine')
                rel_sim_sort = rel_sim_rank.argsort()

                rel_simest = rel_sim_sort[:, 0]  # 先只搞一个试试
                rel_simest_ids = r_entity_input_seq[rel_simest]
                rel_simest_feas = gcn_ex[rel_simest_ids]

                i_sim_matrix = scipy.spatial.distance.cdist(l_ent_i_em, rel_simest_feas, metric='cosine')
                i_sim_diag = np.einsum("ii->i", i_sim_matrix)
                i_sim_score = np.mean(i_sim_diag)
                # align_candidate.append(i_sim_score)
                align_candidate[rank_i] = i_sim_score

        if len(source_output_seq) != 0:

            l_relation_output_seq = source_output_seq[:, 0]
            l_entity_output_seq = source_output_seq[:, 1]

            l_rel_o_em = gcn_rx[l_relation_output_seq]
            l_ent_o_em = gcn_ex[l_entity_output_seq]

            for rank_i, sim_aligns in enumerate(top_rank):
            # for sim_aligns in top_rank:
                this_ent_id = r_ent_ids[sim_aligns]
                target_output_seq = np.array(r_test_o[this_ent_id])
                # target_output_seq = np.array(r_test_o[this_ent_id])
                if len(target_output_seq) == 0:
                    align_candidate[rank_i] += 2
                    continue

                r_relation_out_seq = target_output_seq[:, 0]
                r_entity_out_seq = target_output_seq[:, 1]
                r_rel_o_em = gcn_rx[r_relation_out_seq]
                # r_ent_o_em = gcn_x[r_entity_out_seq]

                # rel_sim_rank = cosine_distance(l_rel_o_em, r_rel_o_em)
                rel_sim_rank = scipy.spatial.distance.cdist(l_rel_o_em, r_rel_o_em, metric='cosine')
                rel_sim_sort = rel_sim_rank.argsort()

                rel_simest = rel_sim_sort[:, 0]  # 先只搞一个试试
                rel_simest_ids = r_entity_out_seq[rel_simest]
                rel_simest_feas = gcn_ex[rel_simest_ids]

                o_sim_matrix = scipy.spatial.distance.cdist(l_ent_o_em, rel_simest_feas, metric='cosine')
                o_sim_diag = np.einsum("ii->i", o_sim_matrix)
                o_sim_score = np.mean(o_sim_diag)

                align_candidate[rank_i] += o_sim_score

        align_rank = np.array(align_candidate).argsort()
        align_index_rank = top_rank[align_rank]

        rank_index = np.where(align_index_rank == i)[0]
        if len(rank_index) == 0:
            rank_index = 50
        else:
            rank_index = rank_index[0]

        left_mrr_rank.append(1 / (rank_index + 1))

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f%%' % (np.mean(left_mrr_rank)))

def nc_cos_get_hits_mrr02(Lvec, Rvec,  gcn_ex, gcn_rx, test_proximity_tup, test_pair, top_k=(1, 10, 50, 100)):
    rcsim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')

    # rcsim = standardization(rcsim)
    sim_k = 20

    l_test_i, l_test_o, r_test_i, r_test_o = test_proximity_tup

    l_ent_ids = list(l_test_i.keys())
    r_ent_ids = list(r_test_i.keys())

    top_lr = [0] * len(top_k)
    left_mrr_rank = []

    for i, entity_id in enumerate(l_test_i):
        rank = rcsim[i, :].argsort()
        top_rank = rank[0: sim_k]
        # sim_rank_dict[i] = top_rank

        source_input_seq = np.array(l_test_i[entity_id])
        source_output_seq = np.array(l_test_o[entity_id])

        align_candidate = np.zeros([len(top_rank)])

        if len(source_input_seq) != 0:
            l_relation_input_seq = source_input_seq[:, 0]
            l_entity_input_seq = source_input_seq[:, 1]

            l_rel_i_em = gcn_rx[l_relation_input_seq]
            l_ent_i_em = gcn_ex[l_entity_input_seq]

            for rank_i, sim_aligns in enumerate(top_rank):
                this_ent_id = r_ent_ids[sim_aligns]
                target_input_seq = np.array(r_test_i[this_ent_id])
                # target_output_seq = np.array(r_test_o[this_ent_id])
                if len(target_input_seq) != 0:
                    r_relation_input_seq = target_input_seq[:, 0]
                    r_entity_input_seq = target_input_seq[:, 1]
                    r_rel_o_em = gcn_rx[r_relation_input_seq]
                    # r_ent_em = gcn_x[r_entity_input_seq]

                    # rel_sim_rank = cosine_distance(l_rel_i_em, r_rel_o_em)
                    rel_sim_rank = scipy.spatial.distance.cdist(l_rel_i_em, r_rel_o_em, metric='cosine')
                    rel_sim_sort = rel_sim_rank.argsort()

                    rel_simest = rel_sim_sort[:, 0]  # 先只搞一个试试
                    rel_simest_ids = r_entity_input_seq[rel_simest]
                    rel_simest_feas = gcn_ex[rel_simest_ids]

                    i_sim_matrix = scipy.spatial.distance.cdist(l_ent_i_em, rel_simest_feas, metric='cosine')
                    i_sim_diag = np.einsum("ii->i", i_sim_matrix)
                    i_sim_score = np.mean(i_sim_diag)
                    # align_candidate.append(i_sim_score)
                    align_candidate[rank_i] = i_sim_score


                align_candidate[rank_i] = 2

        if len(source_output_seq) != 0:

            l_relation_output_seq = source_output_seq[:, 0]
            l_entity_output_seq = source_output_seq[:, 1]

            l_rel_o_em = gcn_rx[l_relation_output_seq]
            l_ent_o_em = gcn_ex[l_entity_output_seq]

            for rank_i, sim_aligns in enumerate(top_rank):
            # for sim_aligns in top_rank:
                this_ent_id = r_ent_ids[sim_aligns]
                target_output_seq = np.array(r_test_o[this_ent_id])
                # target_output_seq = np.array(r_test_o[this_ent_id])
                if len(target_output_seq) == 0:
                    align_candidate[rank_i] += 2
                    continue

                r_relation_out_seq = target_output_seq[:, 0]
                r_entity_out_seq = target_output_seq[:, 1]
                r_rel_o_em = gcn_rx[r_relation_out_seq]
                # r_ent_o_em = gcn_x[r_entity_out_seq]

                # rel_sim_rank = cosine_distance(l_rel_o_em, r_rel_o_em)
                rel_sim_rank = scipy.spatial.distance.cdist(l_rel_o_em, r_rel_o_em, metric='cosine')
                rel_sim_sort = rel_sim_rank.argsort()

                rel_simest = rel_sim_sort[:, 0]  # 先只搞一个试试
                rel_simest_ids = r_entity_out_seq[rel_simest]
                rel_simest_feas = gcn_ex[rel_simest_ids]

                o_sim_matrix = scipy.spatial.distance.cdist(l_ent_o_em, rel_simest_feas, metric='cosine')
                o_sim_diag = np.einsum("ii->i", o_sim_matrix)
                o_sim_score = np.mean(o_sim_diag)

                align_candidate[rank_i] += o_sim_score

        align_rank = np.array(align_candidate).argsort()
        align_index_rank = top_rank[align_rank]

        rank_index = np.where(align_index_rank == i)[0]
        if len(rank_index) == 0:
            rank_index = 50
        else:
            rank_index = rank_index[0]

        left_mrr_rank.append(1 / (rank_index + 1))

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f%%' % (np.mean(left_mrr_rank)))

def nc_cos_get_hits_mrr_torch(Lvec, Rvec, gcn_ex, gcn_rx, test_proximity_tup, test_pair, top_k=(1, 10, 50, 100)):
    # rcsim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')

    sim_k = 20
    rel_select = 0.9
    gamma1 = 0.9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rcsim = cosine_distance(Lvec, Rvec)
    # sim_rank_sort = torch.argsort(rcsim, dim=1)
    # sim_score_sort = torch.sort(rcsim, dim=1)
    sim_sorted, sim_ranks = torch.sort(rcsim, dim=1)
    sim_top_sorted = sim_sorted[:, -sim_k:]
    sim_top_ranks = sim_ranks[:, -sim_k:]

    # rcsim = standardization(rcsim)
    l_test_i, l_test_o, r_test_i, r_test_o = test_proximity_tup

    l_ent_ids = list(l_test_i.keys())
    r_ent_ids = list(r_test_i.keys())

    top_lr = [0] * len(top_k)
    # left_mrr_rank = []
    left_mrr_rank = torch.zeros(len(l_ent_ids), 1, device=device)

    # target_em_dict = {}
    # for j, entity_id in enumerate(r_ent_ids):
    #     target_em_dict[entity_id] =


    for i, entity_id in enumerate(l_ent_ids):
        top_rank = sim_top_ranks[i]

        source_input_seq = l_test_i[entity_id]
        source_input_seq = torch.tensor(source_input_seq, dtype=torch.long).to(device)
        # source_input_seq = torch.tensor(source_input_seq, dtype=torch.long)
        source_output_seq = l_test_o[entity_id]
        source_output_seq = torch.tensor(source_output_seq, dtype=torch.long).to(device)
        # source_output_seq = torch.tensor(source_output_seq, dtype=torch.long)

        if len(source_input_seq) != 0:
            l_relation_input_seq = source_input_seq[:, 0]
            l_entity_input_seq = source_input_seq[:, 1]

            source_rel_i_em = torch.index_select(gcn_rx, 0, l_relation_input_seq)
            source_ent_i_em = torch.index_select(gcn_ex, 0, l_entity_input_seq)

            for rank_i, sim_aligns in enumerate(top_rank):
                this_ent_id = r_ent_ids[sim_aligns]
                target_input_seq = r_test_i[this_ent_id]
                target_input_seq = torch.tensor(target_input_seq, dtype=torch.long).to(device)
                # target_input_seq = torch.tensor(target_input_seq, dtype=torch.long)

                if len(target_input_seq) != 0:
                    r_relation_input_seq = target_input_seq[:, 0]
                    r_entity_input_seq = target_input_seq[:, 1]
                    target_rel_i_em = gcn_rx[r_relation_input_seq]
                    target_ent_i_em = gcn_ex[r_entity_input_seq]

                    pos_i_rel_sim = cosine_distance(source_rel_i_em, target_rel_i_em)
                    pos_i_ent_sim = cosine_distance(source_ent_i_em, target_ent_i_em)

                    # sim_select_T = torch.tensor([True]).to(device).repeat(pos_i_rel_sim.shape)
                    # # sim_select_T = torch.tensor([True]).repeat(pos_i_rel_sim.shape)
                    # sim_select_F = torch.tensor([False]).to(device).repeat(pos_i_rel_sim.shape)
                    # # sim_select_F = torch.tensor([False]).repeat(pos_i_rel_sim.shape)
                    #
                    # pos_i_rel_sim_cuda = pos_i_rel_sim.to(device)
                    # pos_i_mask = torch.where(pos_i_rel_sim_cuda > rel_select, sim_select_T, sim_select_F)
                    # # pos_i_mask = pos_i_mask.cpu()
                    #
                    # pos_i_score = torch.masked_select(pos_i_ent_sim, pos_i_mask)
                    # pos_clamp = torch.clamp(pos_i_score, min=gamma1)
                    # i_nc_sim_score = torch.mean(pos_clamp) - gamma1
                    # sim_top_sorted[rank_i] += i_nc_sim_score

                    pos_select_scores, pos_select_indices = torch.sort(pos_i_rel_sim, dim=1)
                    pos_top_select_scores = pos_select_scores[:, 0]
                    pos_top_select_indices = pos_select_indices[:, 0]

                    sim_select_T = torch.tensor([True]).to(device).repeat(pos_top_select_indices.shape)
                    sim_select_F = torch.tensor([False]).to(device).repeat(pos_top_select_indices.shape)

                    pos_i_mask = torch.where(pos_top_select_scores > rel_select, sim_select_T, sim_select_F)

                    pos_i_ent_sim_scores_raw = torch.index_select(pos_i_ent_sim, 1, pos_top_select_indices)
                    pos_i_ent_sim_scores = torch.einsum("ii->i", pos_i_ent_sim_scores_raw)

                    pos_i_score = torch.masked_select(pos_i_ent_sim_scores, pos_i_mask)
                    i_nc_sim_score = torch.mean(pos_i_score)
                    # pos_clamp = torch.clamp(pos_i_score, max=gamma1)
                    # pos_part += torch.sum(pos_clamp) - pos_clamp.size()[0] * gamma1
                    sim_top_sorted[rank_i] += i_nc_sim_score * 0.1

        if len(source_output_seq) != 0:
            l_relation_output_seq = source_output_seq[:, 0]
            l_entity_output_seq = source_output_seq[:, 1]

            source_rel_o_em = torch.index_select(gcn_rx, 0, l_relation_output_seq)
            source_ent_o_em = torch.index_select(gcn_ex, 0, l_entity_output_seq)

            for rank_i, sim_aligns in enumerate(top_rank):
                this_ent_id = r_ent_ids[sim_aligns]
                target_output_seq = r_test_o[this_ent_id]
                target_output_seq = torch.tensor(target_output_seq, dtype=torch.long).to(device)
                # target_output_seq = torch.tensor(target_output_seq, dtype=torch.long)

                if len(target_output_seq) != 0:
                    r_relation_output_seq = target_output_seq[:, 0]
                    r_entity_output_seq = target_output_seq[:, 1]
                    target_rel_o_em = gcn_rx[r_relation_output_seq]
                    target_ent_o_em = gcn_ex[r_entity_output_seq]

                    pos_o_rel_sim = cosine_distance(source_rel_o_em, target_rel_o_em)
                    pos_o_ent_sim = cosine_distance(source_ent_o_em, target_ent_o_em)

                    # sim_select_T = torch.tensor([True]).to(device).repeat(pos_o_rel_sim.shape)
                    # # sim_select_T = torch.tensor([True]).repeat(pos_o_rel_sim.shape)
                    # sim_select_F = torch.tensor([False]).to(device).repeat(pos_o_rel_sim.shape)
                    # # sim_select_F = torch.tensor([False]).repeat(pos_o_rel_sim.shape)
                    #
                    # pos_o_rel_sim_cuda = pos_o_rel_sim.to(device)
                    # pos_o_mask = torch.where(pos_o_rel_sim_cuda > rel_select, sim_select_T, sim_select_F)
                    # # pos_o_mask = pos_o_mask.cpu()
                    # # pos_o_mask = torch.where(pos_o_rel_sim > rel_select, sim_select_T, sim_select_F)
                    # pos_o_score = torch.masked_select(pos_o_ent_sim, pos_o_mask)
                    # pos_clamp = torch.clamp(pos_o_score, min=gamma1)
                    # o_nc_sim_score = torch.mean(pos_clamp) - gamma1
                    # sim_top_sorted[rank_i] += o_nc_sim_score

                    pos_select_scores, pos_select_indices = torch.sort(pos_o_rel_sim, dim=1)
                    pos_top_select_scores = pos_select_scores[:, 0]
                    pos_top_select_indices = pos_select_indices[:, 0]

                    sim_select_T = torch.tensor([True]).to(device).repeat(pos_top_select_indices.shape)
                    sim_select_F = torch.tensor([False]).to(device).repeat(pos_top_select_indices.shape)

                    pos_i_mask = torch.where(pos_top_select_scores > rel_select, sim_select_T, sim_select_F)

                    pos_i_ent_sim_scores_raw = torch.index_select(pos_o_ent_sim, 1, pos_top_select_indices)
                    pos_i_ent_sim_scores = torch.einsum("ii->i", pos_i_ent_sim_scores_raw)

                    pos_i_score = torch.masked_select(pos_i_ent_sim_scores, pos_i_mask)
                    i_nc_sim_score = torch.mean(pos_i_score)
                    # pos_clamp = torch.clamp(pos_i_score, max=gamma1)
                    # pos_part += torch.sum(pos_clamp) - pos_clamp.size()[0] * gamma1
                    sim_top_sorted[rank_i] += i_nc_sim_score * 0.1

        # align_rank = np.array(align_candidate).argsort()
        align_rank = torch.argsort(sim_top_sorted[rank_i], descending=True) # 重新排序
        # align_rank_indices = sim_top_ranks[align_rank] # 找回原indices
        align_rank_indices = top_rank[align_rank] # 找回原indices

        rank_index = (align_rank_indices == i).nonzero()
        if len(rank_index) == 0:
            rank_index = 50
        else:
            rank_index = rank_index[0][0]

        # left_mrr_rank.append(1 / (rank_index + 1))
        left_mrr_rank[i] = 1 / (rank_index + 1)

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1


    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f%%' % (torch.mean(left_mrr_rank)))

def nc_cos_get_hits_mrr_torch02(Lvec, Rvec, gcn_ex, gcn_rx, test_proximity_tup, test_pair, top_k=(1, 10, 50, 100)):
    # rcsim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')

    sim_k = 20
    rel_select = 0.9
    gamma1 = 0.9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rcsim = cosine_distance(Lvec, Rvec)
    # sim_rank_sort = torch.argsort(rcsim, dim=1)
    # sim_score_sort = torch.sort(rcsim, dim=1)
    sim_sorted, sim_ranks = torch.sort(rcsim, dim=1)
    sim_top_sorted = sim_sorted[:, -sim_k:]
    sim_top_ranks = sim_ranks[:, -sim_k:]

    # rcsim = standardization(rcsim)
    l_test_i, l_test_o, r_test_i, r_test_o = test_proximity_tup

    l_ent_ids = list(l_test_i.keys())
    r_ent_ids = list(r_test_i.keys())

    top_lr = [0] * len(top_k)
    left_mrr_rank = []

    # target_em_dict = {}
    # for j, entity_id in enumerate(r_ent_ids):
    #     target_em_dict[entity_id] =


    for i, entity_id in enumerate(l_ent_ids):
        top_rank = sim_top_ranks[i]

        source_input_seq = l_test_i[entity_id]
        source_input_seq = torch.tensor(source_input_seq, dtype=torch.long).to(device)
        # source_input_seq = torch.tensor(source_input_seq, dtype=torch.long)
        source_output_seq = l_test_o[entity_id]
        source_output_seq = torch.tensor(source_output_seq, dtype=torch.long).to(device)
        # source_output_seq = torch.tensor(source_output_seq, dtype=torch.long)

        if len(source_input_seq) != 0:
            l_relation_input_seq = source_input_seq[:, 0]
            l_entity_input_seq = source_input_seq[:, 1]

            source_rel_i_em = torch.index_select(gcn_rx, 0, l_relation_input_seq)
            source_ent_i_em = torch.index_select(gcn_ex, 0, l_entity_input_seq)

            for rank_i, sim_aligns in enumerate(top_rank):
                this_ent_id = r_ent_ids[sim_aligns]
                target_input_seq = r_test_i[this_ent_id]
                target_input_seq = torch.tensor(target_input_seq, dtype=torch.long).to(device)
                # target_input_seq = torch.tensor(target_input_seq, dtype=torch.long)

                if len(target_input_seq) != 0:
                    r_relation_input_seq = target_input_seq[:, 0]
                    r_entity_input_seq = target_input_seq[:, 1]
                    target_rel_i_em = gcn_rx[r_relation_input_seq]
                    target_ent_i_em = gcn_ex[r_entity_input_seq]

                    pos_i_rel_sim = cosine_distance(source_rel_i_em, target_rel_i_em)
                    pos_i_ent_sim = cosine_distance(source_ent_i_em, target_ent_i_em)

                    sim_select_T = torch.tensor([True]).to(device).repeat(pos_i_rel_sim.shape)
                    # sim_select_T = torch.tensor([True]).repeat(pos_i_rel_sim.shape)
                    sim_select_F = torch.tensor([False]).to(device).repeat(pos_i_rel_sim.shape)
                    # sim_select_F = torch.tensor([False]).repeat(pos_i_rel_sim.shape)

                    pos_i_rel_sim_cuda = pos_i_rel_sim.to(device)
                    pos_i_mask = torch.where(pos_i_rel_sim_cuda > rel_select, sim_select_T, sim_select_F)
                    # pos_i_mask = pos_i_mask.cpu()

                    pos_i_score = torch.masked_select(pos_i_ent_sim, pos_i_mask)
                    pos_clamp = torch.clamp(pos_i_score, min=gamma1)
                    i_nc_sim_score = torch.mean(pos_clamp) - gamma1
                    sim_top_sorted[rank_i] += i_nc_sim_score

        if len(source_output_seq) != 0:
            l_relation_output_seq = source_output_seq[:, 0]
            l_entity_output_seq = source_output_seq[:, 1]

            source_rel_o_em = torch.index_select(gcn_rx, 0, l_relation_output_seq)
            source_ent_o_em = torch.index_select(gcn_ex, 0, l_entity_output_seq)

            for rank_i, sim_aligns in enumerate(top_rank):
                this_ent_id = r_ent_ids[sim_aligns]
                target_output_seq = r_test_o[this_ent_id]
                target_output_seq = torch.tensor(target_output_seq, dtype=torch.long).to(device)
                # target_output_seq = torch.tensor(target_output_seq, dtype=torch.long)

                if len(target_output_seq) != 0:
                    r_relation_output_seq = target_output_seq[:, 0]
                    r_entity_output_seq = target_output_seq[:, 1]
                    target_rel_o_em = gcn_rx[r_relation_output_seq]
                    target_ent_o_em = gcn_ex[r_entity_output_seq]

                    pos_o_rel_sim = cosine_distance(source_rel_o_em, target_rel_o_em)
                    pos_o_ent_sim = cosine_distance(source_ent_o_em, target_ent_o_em)

                    sim_select_T = torch.tensor([True]).to(device).repeat(pos_o_rel_sim.shape)
                    # sim_select_T = torch.tensor([True]).repeat(pos_o_rel_sim.shape)
                    sim_select_F = torch.tensor([False]).to(device).repeat(pos_o_rel_sim.shape)
                    # sim_select_F = torch.tensor([False]).repeat(pos_o_rel_sim.shape)

                    pos_o_rel_sim_cuda = pos_o_rel_sim.to(device)
                    pos_o_mask = torch.where(pos_o_rel_sim_cuda > rel_select, sim_select_T, sim_select_F)
                    # pos_o_mask = pos_o_mask.cpu()
                    # pos_o_mask = torch.where(pos_o_rel_sim > rel_select, sim_select_T, sim_select_F)
                    pos_o_score = torch.masked_select(pos_o_ent_sim, pos_o_mask)
                    pos_clamp = torch.clamp(pos_o_score, min=gamma1)
                    o_nc_sim_score = torch.mean(pos_clamp) - gamma1
                    sim_top_sorted[rank_i] += o_nc_sim_score

        # align_rank = np.array(align_candidate).argsort()
        align_rank = torch.argsort(sim_top_sorted, dim=1, descending=True) # 重新排序
        align_rank_indices = sim_ranks[align_rank] # 找回原indices

        rank_index = (align_rank_indices == i).nonzero()
        if len(rank_index) == 0:
            rank_index = 50
        else:
            rank_index = rank_index[0][0]

        left_mrr_rank.append(1 / (rank_index + 1))

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f%%' % (np.mean(left_mrr_rank)))

def nc_cos_get_hits_mrr_np(Lvec, Rvec, gcn_ex, gcn_rx, test_proximity_tup, test_pair, top_k=(1, 10, 50, 100)):
    # rcsim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')

    sim_k = 20
    rel_select = 0.1
    ent_sim_dis = 0.1

    # rcsim = cosine_distance(Lvec, Rvec)
    rcsim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    # sim_rank_sort = torch.argsort(rcsim, dim=1)
    # sim_score_sort = torch.sort(rcsim, dim=1)
    sim_sorted = np.sort(rcsim, axis=1)
    sim_ranks = np.argsort(rcsim, axis=1)
    sim_top_sorted = sim_sorted[:, :sim_k]
    sim_top_ranks = sim_ranks[:, :sim_k]

    # rcsim = standardization(rcsim)
    l_test_i, l_test_o, r_test_i, r_test_o = test_proximity_tup

    l_ent_ids = list(l_test_i.keys())
    r_ent_ids = list(r_test_i.keys())

    top_lr = [0] * len(top_k)
    left_mrr_rank = []

    for i, entity_id in enumerate(l_ent_ids):
        top_rank = sim_top_ranks[i]

        source_input_seq = l_test_i[entity_id]
        source_input_seq = np.array(source_input_seq)
        source_output_seq = l_test_o[entity_id]
        source_output_seq = np.array(source_output_seq)

        if len(source_input_seq) != 0:
            l_relation_input_seq = source_input_seq[:, 0]
            l_entity_input_seq = source_input_seq[:, 1]

            # source_rel_i_em = torch.index_select(gcn_rx, 0, l_relation_input_seq)
            source_rel_i_em = gcn_rx[l_relation_input_seq]
            # source_ent_i_em = torch.index_select(gcn_ex, 0, l_entity_input_seq)
            source_ent_i_em = gcn_ex[l_entity_input_seq]

            for rank_i, sim_aligns in enumerate(top_rank):
                this_ent_id = r_ent_ids[sim_aligns]
                target_input_seq = r_test_i[this_ent_id]
                # target_input_seq = torch.tensor(target_input_seq, dtype=torch.long).to(device)
                # target_input_seq = torch.tensor(target_input_seq, dtype=torch.long)
                target_input_seq = np.array(target_input_seq)

                if len(target_input_seq) != 0:
                    r_relation_input_seq = target_input_seq[:, 0]
                    r_entity_input_seq = target_input_seq[:, 1]
                    target_rel_i_em = gcn_rx[r_relation_input_seq]
                    target_ent_i_em = gcn_ex[r_entity_input_seq]

                    # pos_i_rel_sim = cosine_distance(source_rel_i_em, target_rel_i_em)
                    pos_i_rel_sim = scipy.spatial.distance.cdist(source_rel_i_em, target_rel_i_em, metric='cosine')
                    # pos_i_ent_sim = cosine_distance(source_ent_i_em, target_ent_i_em)
                    pos_i_ent_sim = scipy.spatial.distance.cdist(source_ent_i_em, target_ent_i_em, metric='cosine')

                    # sim_select_T = torch.tensor([True]).to(device).repeat(pos_i_rel_sim.shape)
                    # sim_select_T = torch.tensor([True]).repeat(pos_i_rel_sim.shape)
                    # sim_select_F = torch.tensor([False]).to(device).repeat(pos_i_rel_sim.shape)
                    # sim_select_F = torch.tensor([False]).repeat(pos_i_rel_sim.shape)

                    # pos_i_rel_sim_cuda = pos_i_rel_sim.to(device)
                    # pos_i_mask = torch.where(pos_i_rel_sim_cuda > rel_select, sim_select_T, sim_select_F)
                    pos_i_select = np.where(pos_i_rel_sim < rel_select)

                    if len(pos_i_select[0]) == 0:
                        continue

                    # pos_i_score = torch.masked_select(pos_i_ent_sim, pos_i_mask)
                    pos_i_score = pos_i_ent_sim[pos_i_select]
                    # pos_clamp = torch.clamp(pos_i_score, min=gamma1)
                    pos_clamp = np.where(pos_i_score < ent_sim_dis)

                    if len(pos_clamp[0]) == 0:
                        continue

                    pos_esim_select = pos_i_score[pos_clamp]

                    i_nc_sim_score = ent_sim_dis - np.mean(pos_esim_select)
                    sim_top_sorted[rank_i] += i_nc_sim_score

        if len(source_output_seq) != 0:
            l_relation_output_seq = source_output_seq[:, 0]
            l_entity_output_seq = source_output_seq[:, 1]

            source_rel_o_em = gcn_rx[l_relation_output_seq]
            # source_rel_o_em = torch.index_select(gcn_rx, 0, l_relation_output_seq)
            source_ent_o_em = gcn_ex[l_entity_output_seq]
            # source_ent_o_em = torch.index_select(gcn_ex, 0, l_entity_output_seq)

            for rank_i, sim_aligns in enumerate(top_rank):
                this_ent_id = r_ent_ids[sim_aligns]
                target_output_seq = r_test_o[this_ent_id]
                # target_output_seq = torch.tensor(target_output_seq, dtype=torch.long).to(device)
                # target_output_seq = torch.tensor(target_output_seq, dtype=torch.long)
                target_output_seq = np.array(target_output_seq)

                if len(target_output_seq) != 0:
                    r_relation_output_seq = target_output_seq[:, 0]
                    r_entity_output_seq = target_output_seq[:, 1]
                    target_rel_o_em = gcn_rx[r_relation_output_seq]
                    target_ent_o_em = gcn_ex[r_entity_output_seq]

                    # pos_o_rel_sim = cosine_distance(source_rel_o_em, target_rel_o_em)
                    pos_o_rel_sim = scipy.spatial.distance.cdist(source_rel_o_em, target_rel_o_em, metric='cosine')
                    # pos_o_ent_sim = cosine_distance(source_ent_o_em, target_ent_o_em)
                    pos_o_ent_sim = scipy.spatial.distance.cdist(source_ent_o_em, target_ent_o_em, metric='cosine')

                    # sim_select_T = torch.tensor([True]).to(device).repeat(pos_o_rel_sim.shape)
                    # sim_select_T = torch.tensor([True]).repeat(pos_o_rel_sim.shape)
                    # sim_select_F = torch.tensor([False]).to(device).repeat(pos_o_rel_sim.shape)
                    # sim_select_F = torch.tensor([False]).repeat(pos_o_rel_sim.shape)

                    pos_o_select = np.where(pos_o_rel_sim < rel_select)

                    if len(pos_o_select[0]) == 0:
                        continue

                    pos_o_score = pos_o_ent_sim[pos_o_select]
                    pos_clamp = np.where(pos_o_score < ent_sim_dis)

                    if len(pos_clamp[0]) == 0:
                        continue

                    pos_esim_select = pos_o_score[pos_clamp]
                    o_nc_sim_score = np.mean(pos_esim_select) - ent_sim_dis

                    # pos_o_rel_sim_cuda = pos_o_rel_sim.to(device)
                    # pos_o_mask = torch.where(pos_o_rel_sim_cuda > rel_select, sim_select_T, sim_select_F)
                    # pos_o_mask = pos_o_mask.cpu()
                    # # pos_o_mask = torch.where(pos_o_rel_sim > rel_select, sim_select_T, sim_select_F)
                    # pos_o_score = torch.masked_select(pos_o_ent_sim, pos_o_mask)
                    # pos_clamp = torch.clamp(pos_o_score, min=gamma1)
                    # o_nc_sim_score = torch.mean(pos_clamp) - gamma1
                    sim_top_sorted[rank_i] += o_nc_sim_score

        # align_rank = np.array(align_candidate).argsort()
        # align_rank = torch.argsort(sim_top_sorted, dim=1, descending=True) # 重新排序
        align_rank = np.argsort(sim_top_sorted, axis=1) # 重新排序
        align_rank_indices = sim_ranks[align_rank] # 找回原indices

        # rank_index = (align_rank_indices == i).nonzero()
        rank_index = np.where(align_rank_indices == i)[0]
        # if rank_index.shape == 0:
        #     rank_index = 50
        # else:
        #     rank_index = rank_index[0][0]

        if len(rank_index) == 0:
            rank_index = 50
        else:
            rank_index = rank_index[0]


        left_mrr_rank.append(1 / (rank_index + 1))

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f%%' % (np.mean(left_mrr_rank)))
    print("*" * 10)

def rel_cos_get_hits_mrr(Lvec, Rvec, test_pair, top_k=(1, 10, 50, 100)):
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    top_lr = [0] * len(top_k)
    left_mrr_rank = []
    # right_mrr_rank = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        left_mrr_rank.append(1 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    # top_rl = [0] * len(top_k)
    # for i in range(Rvec.shape[0]):
    #     rank = sim[:, i].argsort()
    #     rank_index = np.where(rank == i)[0][0]
    #
    #     right_mrr_rank.append(1 / (rank_index + 1))
    #
    #     for j in range(len(top_k)):
    #         if rank_index < top_k[j]:
    #             top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Relation Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('Relation MRR: %.4f%%' % (np.mean(left_mrr_rank)))

    # print('For each right:')
    # for i in range(len(top_rl)):
    #     print('Hits@%d: %.4f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    # print('MRR: %.4f%%' % (np.mean(right_mrr_rank)))

def ent_cos_get_hits_mrr_f(Lvec, Rvec, test_pair, log_f, top_k=(1, 10, 50, 100)):
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    top_lr = [0] * len(top_k)
    left_mrr_rank = []
    # right_mrr_rank = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        left_mrr_rank.append(1 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    # top_rl = [0] * len(top_k)
    # for i in range(Rvec.shape[0]):
    #     rank = sim[:, i].argsort()
    #     rank_index = np.where(rank == i)[0][0]
    #
    #     right_mrr_rank.append(1 / (rank_index + 1))
    #
    #     for j in range(len(top_k)):
    #         if rank_index < top_k[j]:
    #             top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Entity Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('Entity MRR: %.4f%%' % (np.mean(left_mrr_rank)))
    print("*"*10)

    # print('For each right:')
    # for i in range(len(top_rl)):
    #     print('Hits@%d: %.4f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    # print('MRR: %.4f%%' % (np.mean(right_mrr_rank)))

def ent_cos_get_hits_mrr(Lvec, Rvec, test_pair, top_k=(1, 10, 50, 100)):
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    top_lr = [0] * len(top_k)
    left_mrr_rank = []
    # right_mrr_rank = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        left_mrr_rank.append(1 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    # top_rl = [0] * len(top_k)
    # for i in range(Rvec.shape[0]):
    #     rank = sim[:, i].argsort()
    #     rank_index = np.where(rank == i)[0][0]
    #
    #     right_mrr_rank.append(1 / (rank_index + 1))
    #
    #     for j in range(len(top_k)):
    #         if rank_index < top_k[j]:
    #             top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Entity Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('Entity MRR: %.4f%%' % (np.mean(left_mrr_rank)))
    print("*"*10)

    # print('For each right:')
    # for i in range(len(top_rl)):
    #     print('Hits@%d: %.4f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    # print('MRR: %.4f%%' % (np.mean(right_mrr_rank)))


def nc_cos_get_hits_mrr_voter(Lvec, Rvec, gcn_ex, gcn_rx, test_proximity_tup, test_pair, top_k=(1, 10, 50, 100)):
    # rcsim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')

    sim_k = 20
    rel_select = 0.9
    gamma1 = 0.9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rcsim = cosine_distance(Lvec, Rvec)
    sim_sorted, sim_ranks = torch.sort(rcsim, dim=1)

    # sim_top_sorted = sim_sorted[:, -sim_k:]
    # sim_top_ranks = sim_ranks[:, -sim_k:]

    # rcsim = standardization(rcsim)
    l_test_i, l_test_o, r_test_i, r_test_o = test_proximity_tup

    l_ent_ids = list(l_test_i.keys())
    r_ent_ids = list(r_test_i.keys())

    top_lr = [0] * len(top_k)
    # left_mrr_rank = []
    left_mrr_rank = torch.zeros(len(l_ent_ids), 1, device=device)

    for i, entity_id in enumerate(l_ent_ids):

        this_rcsim = rcsim[i]
        need_vote_indices = (this_rcsim >= gamma1).nonzero()
        # need_vote_values =
        if len(need_vote_indices) > 1:
            source_input_seq = l_test_i[entity_id]
            source_input_seq = torch.tensor(source_input_seq, dtype=torch.long).to(device)

            source_output_seq = l_test_o[entity_id]
            source_output_seq = torch.tensor(source_output_seq, dtype=torch.long).to(device)

            vote_values = torch.zeros(len(need_vote_indices), device=device)

            if len(source_input_seq) != 0:
                l_relation_input_seq = source_input_seq[:, 0]
                l_entity_input_seq = source_input_seq[:, 1]

                source_rel_i_em = torch.index_select(gcn_rx, 0, l_relation_input_seq)
                source_ent_i_em = torch.index_select(gcn_ex, 0, l_entity_input_seq)

                # for rank_i, sim_aligns in enumerate(top_rank):

                for rank_i, this_index in enumerate(need_vote_indices):
                    this_ent_id = r_ent_ids[this_index]
                    target_input_seq = r_test_i[this_ent_id]
                    target_input_seq = torch.tensor(target_input_seq, dtype=torch.long).to(device)
                    # target_input_seq = torch.tensor(target_input_seq, dtype=torch.long)

                    if len(target_input_seq) != 0:
                        r_relation_input_seq = target_input_seq[:, 0]
                        r_entity_input_seq = target_input_seq[:, 1]
                        target_rel_i_em = gcn_rx[r_relation_input_seq]
                        target_ent_i_em = gcn_ex[r_entity_input_seq]

                        pos_i_rel_sim = cosine_distance(source_rel_i_em, target_rel_i_em)
                        pos_i_ent_sim = cosine_distance(source_ent_i_em, target_ent_i_em)

                        pos_select_scores, pos_select_indices = torch.sort(pos_i_rel_sim, dim=1)
                        pos_top_select_scores = pos_select_scores[:, 0]
                        pos_top_select_indices = pos_select_indices[:, 0]

                        sim_select_T = torch.tensor([True]).to(device).repeat(pos_top_select_indices.shape)
                        sim_select_F = torch.tensor([False]).to(device).repeat(pos_top_select_indices.shape)

                        pos_i_mask = torch.where(pos_top_select_scores > rel_select, sim_select_T, sim_select_F)

                        pos_i_ent_sim_scores_raw = torch.index_select(pos_i_ent_sim, 1, pos_top_select_indices)
                        pos_i_ent_sim_scores = torch.einsum("ii->i", pos_i_ent_sim_scores_raw)

                        pos_i_score = torch.masked_select(pos_i_ent_sim_scores, pos_i_mask)
                        i_nc_sim_score = torch.mean(pos_i_score)
                        vote_values[rank_i] = i_nc_sim_score

            if len(source_output_seq) != 0:
                l_relation_output_seq = source_output_seq[:, 0]
                l_entity_output_seq = source_output_seq[:, 1]

                source_rel_o_em = torch.index_select(gcn_rx, 0, l_relation_output_seq)
                source_ent_o_em = torch.index_select(gcn_ex, 0, l_entity_output_seq)

                for rank_i, sim_aligns in enumerate(need_vote_indices):
                    this_ent_id = r_ent_ids[sim_aligns]
                    target_output_seq = r_test_o[this_ent_id]
                    target_output_seq = torch.tensor(target_output_seq, dtype=torch.long).to(device)
                    # target_output_seq = torch.tensor(target_output_seq, dtype=torch.long)

                    if len(target_output_seq) != 0:
                        r_relation_output_seq = target_output_seq[:, 0]
                        r_entity_output_seq = target_output_seq[:, 1]
                        target_rel_o_em = gcn_rx[r_relation_output_seq]
                        target_ent_o_em = gcn_ex[r_entity_output_seq]

                        pos_o_rel_sim = cosine_distance(source_rel_o_em, target_rel_o_em)
                        pos_o_ent_sim = cosine_distance(source_ent_o_em, target_ent_o_em)

                        pos_select_scores, pos_select_indices = torch.sort(pos_o_rel_sim, dim=1)
                        pos_top_select_scores = pos_select_scores[:, 0]
                        pos_top_select_indices = pos_select_indices[:, 0]

                        sim_select_T = torch.tensor([True]).to(device).repeat(pos_top_select_indices.shape)
                        sim_select_F = torch.tensor([False]).to(device).repeat(pos_top_select_indices.shape)

                        pos_i_mask = torch.where(pos_top_select_scores > rel_select, sim_select_T, sim_select_F)

                        pos_i_ent_sim_scores_raw = torch.index_select(pos_o_ent_sim, 1, pos_top_select_indices)
                        pos_i_ent_sim_scores = torch.einsum("ii->i", pos_i_ent_sim_scores_raw)

                        pos_i_score = torch.masked_select(pos_i_ent_sim_scores, pos_i_mask)
                        i_nc_sim_score = torch.sum(pos_i_score)
                        # pos_clamp = torch.clamp(pos_i_score, max=gamma1)
                        # pos_part += torch.sum(pos_clamp) - pos_clamp.size()[0] * gamma1
                        vote_values[rank_i] += i_nc_sim_score

            ranc_index = torch.argsort(vote_values)[-1]
            ranc_source_index = need_vote_indices[ranc_index]

            # left_mrr_rank.append(1 / (ranc_source_index + 1))
            left_mrr_rank[i] = 1 / (ranc_source_index + 1)

            for j in range(len(top_k)):
                if ranc_source_index < top_k[j]:
                    top_lr[j] += 1

        else:
            # if len(need_vote_indices) == 0:
            #     rank_index = np.where(this_rcsim == i)[0][0]
            # else:
            #     rank_index = need_vote_indices[0][0]
            if len(need_vote_indices) == 0:
                rank_index = (this_rcsim == i).nonzero()
            # if len(rank_index) == 0:
            #     rank_index = 50
            else:
                rank_index = need_vote_indices[0][0]


            left_mrr_rank[i] = 1 / (rank_index + 1)

            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1

    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f%%' % (torch.mean(left_mrr_rank)))



def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

