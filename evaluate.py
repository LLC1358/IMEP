import numpy as np

from sklearn.metrics import roc_auc_score

###################################################################################################################################################

def mrr_score(y_true, y_pred):
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    
    return np.sum(rr_score) / np.sum(y_true)

###############################################################

def dcg_score(y_true, y_pred, k):
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    
    return np.sum(gains / discounts)

###############################################################

def ndcg_score(y_true, y_pred, k):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_pred, k)
    
    return actual / best

###################################################################################################################################################

def evaluate(predicts, truths):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    for pred, true in zip(predicts, truths):
        y_true = np.array(true, dtype='float32')
        y_pred = 1.0 / np.array(pred, dtype='float32')

        auc = roc_auc_score(y_true, y_pred)
        mrr = mrr_score(y_true, y_pred)
        ndcg5 = ndcg_score(y_true, y_pred, 5)
        ndcg10 = ndcg_score(y_true, y_pred, 10)

        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)

###################################################################################################################################################

'''
y_true = [0, 0, 1, 0]
y_pred = [3, 1, 2, 4]   # rank
1 / y_pred = [0.33, 1, 0.5, 0.25]

### AUC ###
0.5 > 0.33
0.5 < 1
0.5 > 0.25
AUC = 2/3

### MRR ###
order = [1, 2, 0, 3]
y_true = [0, 1, 0, 0]
rr_score  = [0/1, 1/2, 0/3, 0/4] = [0, 0.5, 0, 0]
mrr = 0.5 / 1 = 0.5

### NDCG@3 ###
order = [1, 2, 0]
actual y_true = [0, 1, 0]
actual gains = 2 ** [0, 1, 0] - 1 = [0, 1, 0]
actual discounts = [log2(2), log2(3), log2(4)] = [1, 1.585, 2]
actual DCG@3 = (0/1 + 1/1.585 + 0/2) = 0.631

order = [2, 0, 1]
best y_true = [1, 0, 0]
best gains = 2 ** [1, 0, 0] - 1 = [1, 0, 0]
best discounts = [log2(2), log2(3), log2(4)] = [1, 1.585, 2]
best DCG@3 = (1/1 + 0/1.585 + 0/2) = 1.0

NDCG@5 = 0.631 / 1.0 = 0.631
'''