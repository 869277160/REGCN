from Config import *
import numpy as np

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

def get_entity_relation_nums(triples):
    entity_ids = []
    relation_ids = []
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        entity_ids.append(head)
        entity_ids.append(tail)
        relation_ids.append(relation)



    entity_ids_set = set(entity_ids)
    relation_ids_set = set(relation_ids)



    return entity_ids_set, relation_ids_set

def data_check_nums(s_KG, t_KG):

    '''
    :param s_KG:
    :param t_KG:
    :return: None
    '''
    s_entities, s_relations = get_entity_relation_nums(s_KG)
    t_entities, t_relations = get_entity_relation_nums(t_KG)


    m_KG = s_KG + t_KG
    m_entities, m_relations = get_entity_relation_nums(m_KG)
    print("m_entity_num:", len(m_entities))  # s_entity_num: 19388
    print("m_relation_num:", len(m_relations))  # s_relation_num: 1701


    '''
    统计KG中的entity和relation数量
    '''
    # print("s_entity_num:", len(s_entities))  # s_entity_num: 19388
    # print("s_relation_num:", len(s_relations))  # s_relation_num: 1701
    # print("t_entity_num:", len(t_entities))  # t_entity_num: 19572
    # print("t_relation_num:", len(t_relations))  # t_relation_num: 1323
    # print("merge_relation_num:", len(s_relations | t_relations))  # merged_relation_num: 3024
    # print("intersect_relation_num:", len(s_relations & t_relations))  # intersect_relation_num: 0


    '''
    找到最大的entity id
    '''
    m_entities = s_entities | t_entities
    m_relations = s_relations | t_relations

    m_entities_sort = list(m_entities)
    m_relations_sort = list(m_relations)

    m_entities_sort.sort()
    m_relations_sort.sort()

    print(m_entities_sort[0], m_entities_sort[-1])  # 0 38959
    print(m_relations_sort[0], m_relations_sort[-1])    # 0 3023

def load_s_entity_info(fn):
    entity_names = {}
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ei = th[0]
            en = th[1].split("http://zh.dbpedia.org/resource/")[1]
            entity_names[ei] = en
    return entity_names

def load_t_entity_info(fn):
    entity_names = {}
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ei = th[0]
            en = th[1].split("http://dbpedia.org/resource/")[1]
            entity_names[ei] = en
    return entity_names

def load_s_relation_info(fn):
    relation_names = {}
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ri = th[0]
            rn = th[1].split("http://zh.dbpedia.org/property/")[1]
            relation_names[ri] = rn
    return relation_names

def load_t_relation_info(fn):
    relation_names = {}
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ri = th[0]
            # en = th[1].split("http://dbpedia.org/property/")[1]
            rn = th[1].split("http://dbpedia.org/property/")[1]
            relation_names[ri] = rn
    return relation_names

def JLER_train_test_set_construction(need_r=False):
    '''
    :return:
    '''
    '''
    构建训练集和测试集
    '''
    entity_cnt = len(set(loadfile(JLER_Config.s_ent_ids, 1)) | set(loadfile(JLER_Config.t_ent_ids, 1)))
    # entity_cnt = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))  # 全部entity数

    entity_list1 = load_entity(JLER_Config.s_ent_ids, 1)
    entity_set1 = list(set(entity_list1))
    # entity_set1 = list(set(load_entity(Config.e1, 1)))

    entity_list2 = load_entity(JLER_Config.t_ent_ids, 1)
    entity_set2 = list(set(entity_list2))
    # entity_set2 = list(set(load_entity(Config.e2, 1)))

    '''
    source KG entity 数量：19388
    target KG entity 数量：19572
    merge KG entity 数量：38960
    '''
    entity_set1.sort(key=entity_list1.index)
    entity_set2.sort(key=entity_list2.index)
    entity_set = entity_set1 + entity_set2

    '''
    ground truth的对齐数量 15000
    '''
    alignment_seeds = loadfile(JLER_Config.ground_truth, 2)
    aligns_num = len(alignment_seeds)
    # np.random.shuffle(alignment_seeds)
    train_aligns = np.array(alignment_seeds[:aligns_num // 10 * JLER_Config.seed])
    test_aligns = alignment_seeds[aligns_num // 10 * JLER_Config.seed:]

    source_KG = loadfile(JLER_Config.s_kg, 3)
    target_KG = loadfile(JLER_Config.t_kg, 3)

    s_entity_names = load_s_entity_info(JLER_Config.s_ent_ids)
    t_entity_names = load_t_entity_info(JLER_Config.t_ent_ids)

    s_relation_names = load_s_relation_info(JLER_Config.s_rel_ids)
    t_relation_names = load_t_relation_info(JLER_Config.t_rel_ids)

    info_dict = [s_entity_names, t_entity_names, s_relation_names, t_relation_names]

    if need_r == True:
        relation_alignments_seeds = loadfile(JLER_Config.r_agt, 2)
        r_aligns_num = len(relation_alignments_seeds)
        r_train_aligns = np.array(relation_alignments_seeds[:r_aligns_num // 10 * JLER_Config.r_seed])
        r_test_aligns = relation_alignments_seeds[r_aligns_num // 10 * JLER_Config.r_seed:]
        # return source_KG, target_KG, train_aligns, test_aligns, r_train_aligns, r_test_aligns, relation_alignments_seeds, \
        #        entity_cnt
        return source_KG, target_KG, train_aligns, test_aligns, r_train_aligns, r_test_aligns, relation_alignments_seeds, info_dict, entity_cnt

    # aligns_np = np.array(alignment_seeds)

    return source_KG, target_KG, train_aligns, test_aligns, entity_cnt


if __name__ == '__main__':

    source_KG = loadfile(JLER_Config.s_kg, 3)
    target_KG = loadfile(JLER_Config.t_kg, 3)


    '''
    1. 统计KG中的entity和relation数量
    2. 找到最大的entity id
    '''
    data_check_nums(source_KG, target_KG)

