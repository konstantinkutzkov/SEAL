import numpy as np
import re
import time
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
# from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from nltk.util import skipgrams
from sklearn.kernel_approximation import RBFSampler, PolynomialCountSketch
from sklearn.neighbors import NearestNeighbors
from utils import *


def random_walk(G, start_node, nr_walks, walk_length, seed=73):
    walks = []
    np.random.seed(seed)
    for _ in range(nr_walks):
        neighbors = G[start_node]
        walk = [start_node]
        for _ in range(walk_length):
            idx = np.random.randint(len(neighbors))
            walk.append(neighbors[idx][0])
            walk.append(neighbors[idx][1])
            #print(neighbors[idx])
            neighbors = G[neighbors[idx][1]]
        walks.append(walk)
    return walks 

def get_corpus(G, nr_walks, walk_length):
    corpus = []
    for node in G.keys():
        corpus.extend(random_walk(G, node, nr_walks=nr_walks, walk_length=walk_length))
    return corpus

def split_entity(s, chars_to_remove=['_', '.', ',', '(', ')', '[', ']', '!']):
    t = re.sub( r"([A-Z]|_)", r" \1", s).split()
    res = ' '.join(t)
    sc = set(chars_to_remove)
    return ''.join([c for c in res if c not in sc])

def get_llm(model_name="https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual/2"): #model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    # llm_model = SentenceTransformer(model_name)
    llm = hub.load(model_name)
    return llm

def get_LLM_embeddings(llm, corpus, emb_dim):
    model_w2v = Word2Vec(vector_size=1, window=5, min_count=1, workers=1)
    model_w2v.build_vocab(corpus)
    vocabulary = sorted(model_w2v.wv.index_to_key)
    sentences = [split_entity(ent) for ent in vocabulary]
    print('Sentences computed')
    embeddings = llm(sentences) #llm_model.encode(sentences)
    pca = PCA(n_components=emb_dim)
    reduced_dim_embeddings = pca.fit_transform(embeddings)
    del embeddings
    assert len(vocabulary) == reduced_dim_embeddings.shape[0]
    print('PCA dim reduction finished')
    emb_dic = {ent: emb for ent, emb in zip(vocabulary, reduced_dim_embeddings)}
    return emb_dic

def train_word2vec(corpus, emb_dim, emb_path, initial_emb_fname, emb_fname):
    model_w2v = Word2Vec(vector_size=emb_dim, window=5, min_count=5, workers=4)
    model_w2v.build_vocab(corpus)
    model_w2v.wv.vectors_lockf = np.ones(len(model_w2v.wv), dtype=np.float32)
    model_w2v.wv.intersect_word2vec_format(emb_path + initial_emb_fname) #"data/embeddings/llm_embeddings.txt")
    model_w2v.wv.vectors_lockf = np.ones(len(model_w2v.wv), dtype=np.float32)
    model_w2v.train(corpus, total_examples=len(corpus), epochs=5)
    model_w2v.wv.save_word2vec_format(emb_path + emb_fname)

def string2vec(string, bins, nr_chars, distance):
    vec = [0 for _ in range(bins)]
    for sg in list(skipgrams(string, nr_chars, distance)):
        vec[hash(''.join(sg))%bins] += 1
    return vec

def attr2vec(attr_map, attr_indices):
    vec = [-1 for _ in range(len(attr_indices)+1)]
    for attr, cnt in attr_map.items():
        vec[attr_indices.get(attr, len(vec)-1)] = cnt
    return vec

def get_vectors(G, embs, ent_attrs, attr_indices, kernel_map, bins, nr_chars=3, distance=1):
    """
        G: knowledge graph as a hash map {head: [(rel, tail), ...]}
        embs: precomputed entity and relations embeddings
        kernel_map: explicit feature map for non-linear kernels, see https://scikit-learn.org/1.5/modules/kernel_approximation.html#kernel-approximation
        bins: the dimensionality of the skip-gram frequency vectors
        nr_chars, distance: skipgram parameters
    """
    X =  np.array(list(embs.values()))
    print(X.shape)
    #sketch = kernel_map.fit_transform(X)
    #print('sketch shape', sketch.shape)
    #assert len(embs) == sketch.shape[0]
    kernel_embs = embs #{entity: kemb for entity, kemb in zip(embs, sketch)}
    cnt = 0
    nf = 0
    entity_kernel_embs = {}
    cnt_attrs = 0
    for node, neighbors in G.items():
        # print(node)
        cnt += 1
        if cnt % 5000 == 0:
            print(cnt)
        emb_head = kernel_embs[node]
        skipgram_head = string2vec(node, bins, nr_chars, distance)
        #nbr_rels, nbr_tails = [], []
        nbrs = []
        skipgram_rels, skipgram_tails = [], []
        degrees = []
        node_attr_vec = [-1 for _ in range(len(attr_indices)+1)]
        if node in ent_attrs:
            node_attrs = ent_attrs[node]
            node_attr_vec = attr2vec(node_attrs, attr_indices)
            cnt_attrs += 1
        for nbr in neighbors:
            if nbr[0] not in embs or nbr[1] not in embs:
                # print('not found', nf, nbr)
                nf += 1
                continue
            emb_rel, emb_tail = np.array(kernel_embs[nbr[0]]), np.array(kernel_embs[nbr[1]])
            emb_rel = emb_rel/np.linalg.norm(emb_rel)
            nbrs.append(np.multiply(emb_rel, emb_tail))
            #nbr_rels.append(emb_rel)
            #nbr_tails.append(emb_tail)
            degrees.append(len(G[nbr[1]]))
            skipgram_rels.append(string2vec(nbr[0], bins, nr_chars, distance))
            skipgram_tails.append(string2vec(nbr[1], bins, nr_chars, distance))
        #emb_rel_agg = np.mean(nbr_rels, axis=0)
        #emb_tails_agg = np.mean(nbr_tails, axis=0)
        emb_nbrs = np.sum(nbrs, axis=0)
        skipgram_rel_agg = np.mean(skipgram_rels, axis=0)
        skipgram_tails_agg = np.mean(skipgram_tails, axis=0)
        v = list(emb_head) + list(emb_nbrs) + list(skipgram_head) + list(skipgram_rel_agg) + list(skipgram_tails_agg) + node_attr_vec
        # v = list(emb_head) + list(emb_rel_agg) + list(emb_tails_agg) + list(skipgram_head) + list(skipgram_rel_agg) + list(skipgram_tails_agg)
        # if len(neighbors) < 1:
        #     print(node, neighbors)
        hist, _ = np.histogram(degrees, bins=[0, 3, 6, 10, 15, 20])
        v.extend(hist)
        v.append(len(neighbors))
        entity_kernel_embs[node] = np.array(v)/np.linalg.norm(v)
    print('#nodes with attributes', cnt_attrs)
    return entity_kernel_embs

if __name__ == '__main__':

    prefixes = ['dummy', 'fr', 'en']
    datapath = f'data/{prefixes[1]}_{prefixes[2]}/'
    nr_walks = 30
    walk_length = 5
    emb_dim = 300
    embeddings_path = 'embeddings/'
    start = time.time()

    print('Generate graphs')

    ent_map1 = read_entities_map(datapath, 'ent_ids_1')
    ent_map2 = read_entities_map(datapath, 'ent_ids_2')

    ent_attr_map1, ent_attr_set1 = get_attribute_frequencies(datapath, f'{prefixes[1]}_att_triples')
    ent_attr_map2, ent_attr_set2 = get_attribute_frequencies(datapath, f'{prefixes[2]}_att_triples')
    attr_set = ent_attr_set1.intersection(ent_attr_set2)

    attr_indices = {attr:i for i, attr in enumerate(sorted(list(attr_set)))}
    
    rel_map1 = read_entities_map(datapath, 'rel_ids_1')
    rel_map2 = read_entities_map(datapath, 'rel_ids_2')

    G1 = get_graph(datapath, 'triples_1', ent_map1, rel_map1)
    G2 = get_graph(datapath, 'triples_2', ent_map2, rel_map2)

    print(len(G1), len(G2))

    print('Generate random walk corpora')
    corpus1 = get_corpus(G1, nr_walks=nr_walks, walk_length=walk_length)
    corpus2 = get_corpus(G2, nr_walks=nr_walks, walk_length=walk_length)

    print('Corpora lengths', len(corpus1), len(corpus2))

    print("Initial LLM embeddings")
    llm = get_llm()
    initial_embeddings1 = get_LLM_embeddings(llm, corpus1, emb_dim=emb_dim)
    initial_embeddings2 = get_LLM_embeddings(llm, corpus2, emb_dim=emb_dim)

    write_to_w2v_format(embeddings_path, f'{prefixes[1]}_llm_embeddings.txt', initial_embeddings1)
    write_to_w2v_format(embeddings_path, f'{prefixes[2]}_llm_embeddings.txt', initial_embeddings2)

    print('DeepWalk training')
    
    train_word2vec(corpus1, emb_dim, embeddings_path, f'{prefixes[1]}_llm_embeddings.txt', f"{prefixes[1]}_deepwalk_embs.txt")
    train_word2vec(corpus2, emb_dim, embeddings_path, f'{prefixes[2]}_llm_embeddings.txt', f"{prefixes[2]}_deepwalk_embs.txt")

    print('Aggregate neighbor embeddings')
    embs1 = load_embeddings(embeddings_path, f"{prefixes[1]}_deepwalk_embs.txt")
    embs2 = load_embeddings(embeddings_path, f"{prefixes[2]}_deepwalk_embs.txt")

    bins = emb_dim
    poly_sketch = None #PolynomialCountSketch(degree=2, n_components=bins, random_state=1) # use the same random state for different graphs
    entity_embs1 = get_vectors(G1, embs1, ent_attr_map1, attr_indices, poly_sketch, bins=bins)
    entity_embs2 = get_vectors(G2, embs2, ent_attr_map2, attr_indices, poly_sketch, bins=bins)

    write_entity_embs_to_file(entity_embs1, embeddings_path, f'{prefixes[1]}_final_embs_{emb_dim}.txt')
    write_entity_embs_to_file(entity_embs2, embeddings_path, f'{prefixes[2]}_final_embs_{emb_dim}.txt')

    print('Elapsed time', time.time()-start)