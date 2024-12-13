import pandas as pd
import numpy as np
import lightgbm as lgb
from utils import *


def entity_pairs(pairs, ent_map2, entity_embs2):
    ents2, embs2 = [], []
    for pair in pairs:
        p2 = ent_map2[pair[1]].split('/')[-1]
        ents2.append(p2)
        embs2.append(entity_embs2[p2])
    return ents2, embs2

def entity_pairs_df(pairs, ent_map2, entity_embs2):
    ents2, embs2 = [], []
    for pair in pairs:
        p2 = ent_map2[pair[1]].split('/')[-1]
        ents2.append(p2)
        embs2.append(entity_embs2[p2])
    return ents2, pd.DataFrame(embs2)

def get_feature_matrix(pair, ent_map1, ent_map2, entity_embs1, ents2, embs2):
    feats = []
    # print(pair)
    p1, p2 = ent_map1[pair[0]].split('/')[-1], ent_map2[pair[1]].split('/')[-1]
    index = ents2.index(p2)
    for emb2 in embs2:
        feats.append(list(entity_embs1[p1]) + list(emb2))
    return pd.DataFrame(feats), index

def get_feature_matrix_fast(pair, ent_map1, ent_map2, entity_embs1, ents2, embs2_df):
    p1, p2 = ent_map1[pair[0]].split('/')[-1], ent_map2[pair[1]].split('/')[-1]
    index = ents2.index(p2)
    df1 = pd.DataFrame(np.array(embs2_df.shape[0]*entity_embs1[p1]).reshape(embs2_df.shape[0], -1))
    res = pd.concat([df1, embs2_df], axis=1)
    return res, index

if __name__ == '__main__':

    prefixes = ['dummy', 'fr', 'en']
    emb_dim = 300
    datapath_embeddings = 'embeddings/'
    entity_embs1 = load_embeddings(datapath_embeddings, f'{prefixes[1]}_final_embs_{emb_dim}.txt')
    entity_embs2 = load_embeddings(datapath_embeddings, f'{prefixes[2]}_final_embs_{emb_dim}.txt')

    datapath = f'data/{prefixes[1]}_{prefixes[2]}/'
    ref_pairs = load_labels(datapath, 'ref_pairs')
   
    ent_map1 = read_entities_map(datapath, 'ent_ids_1')
    ent_map2 = read_entities_map(datapath, 'ent_ids_2')

    lgb_model = lgb.Booster(model_file=f"models/{prefixes[1]}_{prefixes[2]}_model_{emb_dim}.txt")

    entities2, embs2_df = entity_pairs_df(ref_pairs, ent_map2, entity_embs2)
    # print(embs2_df.shape)
    # get_feature_matrix_fast(ref_pairs[0], ent_map1, ent_map2, entity_embs1, entities2, embs2_df)

    # ents2, embs2 = entity_pairs(ref_pairs, ent_map2, entity_embs2)

    top_k = 10

    fnd = 0
    top = 10000
    np.random.seed(42)
    samples = np.random.choice(range(len(ref_pairs)), min(top, len(ref_pairs)), replace=False)
    for i, sample_idx in enumerate(samples): #query in enumerate(range(top)):
        all_feats, index = get_feature_matrix_fast(ref_pairs[sample_idx], ent_map1, ent_map2, entity_embs1, entities2, embs2_df)
        #print(all_feats.shape)
        preds = lgb_model.predict(all_feats)
        if index in preds.argsort()[-top_k:][::-1]:
            fnd += 1
        if (i+1) % 10 == 0:
            print(i+1, np.round(fnd/(i+1), 3))

    print('hits', fnd/top)


