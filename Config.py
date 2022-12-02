'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-12-02 15:15:40
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-12-02 15:21:30
FilePath: /REGCN/Config.py
Description: 

Copyright (c) 2022 by error: git config user.name && git config user.email & please set dead value or install git, All Rights Reserved. 
'''
import torch
import torch.nn as nn



class JLER_Config:
    language = 'zh_en' # zh_en | ja_en | fr_en
    pwd = './data/JLER/'

    s_ent_ids = pwd + language + '/ent_ids_1'
    t_ent_ids = pwd + language + '/ent_ids_2'

    s_rel_ids = pwd + language + '/rel_ids_1'
    t_rel_ids = pwd + language + '/rel_ids_2'

    ground_truth = pwd + language + '/ref_ent_ids'
    r_agt = pwd + language + '/ref_r_ids' # relation
    s_kg = pwd + language + '/triples_1'
    t_kg = pwd + language + '/triples_2'
    epochs = 800
    batch_size = 500
    dim = 300
    act_func = torch.relu
    alpha = 0.1
    beta = 0.3
    gamma = 1.0  # margin based loss
    k = 125  # number of negative samples for each positive one
    seed = 3  # 30% of seeds
    r_seed = 3

    # mo added
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
