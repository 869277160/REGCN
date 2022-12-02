import numpy as np
import scipy.sparse as sp
import scipy

def cos_get_hits_mrr(Lvec, Rvec, test_pair, top_k=(1, 10, 50, 100)):
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    top_lr = [0] * len(top_k)
    left_mrr_rank = []
    right_mrr_rank = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]

        left_mrr_rank.append(1/(rank_index+1))

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
