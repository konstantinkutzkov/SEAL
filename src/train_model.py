import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *



def pairs_to_X_y(pairs, ent_map1, ent_map2, entity_embs1, entity_embs2, nr_neg=5):
    feats, y = [], []
    for pair in pairs:
        p1, p2 = ent_map1[pair[0]].split('/')[-1], ent_map2[pair[1]].split('/')[-1]
        feats.append(list(entity_embs1[p1]) + list(entity_embs2[p2]))
        y.append(1)

    if nr_neg > 0:
        vals1, vals2 = list(entity_embs1.values()), list(entity_embs2.values())
        rs1, rs2 = np.random.choice(list(range(len(vals1))), size=nr_neg*len(y)), np.random.choice(list(range(len(vals2))), size=nr_neg*len(y))

        for idx1, idx2 in zip(rs1, rs2):
            feats.append(list(vals1[idx1]) + list(vals2[idx2]))
            y.append(0)
    assert len(feats) == len(y)
    X = pd.DataFrame(np.array(feats))
    return X, y


def lgb_model(X_train, X_val, y_train, y_val):

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
        params, lgb_train, num_boost_round=6000, valid_sets=lgb_eval, 
        callbacks=[lgb.early_stopping(stopping_rounds=500), lgb.log_evaluation(period=100, show_stdv=True)]
    )
    return gbm

if __name__ == '__main__':

    prefixes = ['dummy', 'fr', 'en']
    emb_dim = 300
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