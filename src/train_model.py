import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from collections import defaultdict
from utils import *


def negative_sampling_naive(p1, entity_embs1, entity_embs2, nr_neg):
    emb1 = entity_embs1[p1]
    entities = list(entity_embs2.keys())
    vals2 = list(entity_embs2.values())
    ns = np.random.choice(list(range(len(vals2))), size=nr_neg)
    feats = []
    for idx in ns:
        feats.append(list(emb1) + list(vals2[idx]))
    return feats

def negative_sampling(p1, entity_embs1, entity_embs2, nbrs_map, k, Y, nbrs):
    emb1 = entity_embs1[p1]
    entities = list(entity_embs2.keys())
    q = np.array(emb1).reshape(1,-1)

    distances, indices = nbrs.kneighbors(q)
    samples = []
    for idx in np.random.permutation(indices[0]):
        # print(idx)
        if (p1, entities[idx]) not in nbrs_map:
            samples.append(list(emb1) + list(entity_embs2[entities[idx]]))
            if len(samples) == k:
                break
    return samples


def get_neighbors(pairs, ent_map1, ent_map2):
    nbrs_map = defaultdict(set)
    for pair in pairs:
        p1, p2 = ent_map1[pair[0]].split('/')[-1], ent_map2[pair[1]].split('/')[-1]
        nbrs_map[p1].add(p2)
    return nbrs_map



def pairs_to_X_y(pairs, ent_map1, ent_map2, entity_embs1, entity_embs2, nr_neg=5):
    nbrs_map = get_neighbors(pairs, ent_map1, ent_map2)
    Y = np.array(list(entity_embs2.values()))
    nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(Y)
    feats, y = [], []
    for i, pair in enumerate(pairs):
        if i%1000 == 0:
            print(i)
        p1, p2 = ent_map1[pair[0]].split('/')[-1], ent_map2[pair[1]].split('/')[-1]
        feats.append(list(entity_embs1[p1]) + list(entity_embs2[p2]))
        y.append(1)

        #neg_samples = negative_sampling(p1, entity_embs1, entity_embs2, nbrs_map, nr_neg, Y, nbrs)
        neg_samples = negative_sampling_naive(p1, entity_embs1, entity_embs2, nr_neg)
        #print(len(neg_samples))
        feats.extend(neg_samples)
        y.extend([0]*len(neg_samples))


    # if nr_neg > 0:
    #     vals1, vals2 = list(entity_embs1.values()), list(entity_embs2.values())
    #     rs1, rs2 = np.random.choice(list(range(len(vals1))), size=nr_neg*len(y)), np.random.choice(list(range(len(vals2))), size=nr_neg*len(y))

    #     for idx1, idx2 in zip(rs1, rs2):
    #         feats.append(list(vals1[idx1]) + list(vals2[idx2]))
    #         y.append(0)
    assert len(feats) == len(y)
    X = pd.DataFrame(np.array(feats))
    return X, y


def lgb_model(X_train, X_val, y_train, y_val, nr_rounds=1000):

    lgb_train = lgb.Dataset(pd.DataFrame(X_train), y_train)
    lgb_eval = lgb.Dataset(pd.DataFrame(X_val), y_val, reference=lgb_train)

    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": {"auc", "average_precision"},
        "num_leaves": 21,
        "min_data_in_leaf": 5,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 1,
        "seed": 73
    }

    gbm = lgb.train(
        params, lgb_train, num_boost_round=nr_rounds, valid_sets=lgb_eval, 
        callbacks=[lgb.early_stopping(stopping_rounds=500), lgb.log_evaluation(period=100, show_stdv=True)]
    )
    return gbm

if __name__ == '__main__':

    prefixes = ['dummy', 'fr', 'en']
    emb_dim = 50
    datapath_embeddings = 'embeddings/'
    entity_embs1 = load_embeddings(datapath_embeddings, f'{prefixes[1]}_final_embs_{emb_dim}.txt')
    entity_embs2 = load_embeddings(datapath_embeddings, f'{prefixes[2]}_final_embs_{emb_dim}.txt')

    datapath = f'data/{prefixes[1]}_{prefixes[2]}/'
    sup_pairs = load_labels(datapath, 'sup_pairs')

    ent_map1 = read_entities_map(datapath, 'ent_ids_1')
    ent_map2 = read_entities_map(datapath, 'ent_ids_2')

    X, y = pairs_to_X_y(sup_pairs, ent_map1, ent_map2,  entity_embs1, entity_embs2, nr_neg=10)
    print('X shape', X.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=73)

    gbm = lgb_model(X_train, X_val, y_train, y_val)

    gbm.save_model(f"models/{prefixes[1]}_{prefixes[2]}_model_{emb_dim}.txt")