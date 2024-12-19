import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
# from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from utils import *
from train_model import *

def page0_main():
    st.markdown("#### Scalable Entity ALignment (SEAL) demo")

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

def load_graph(datasets):
    ent_map, ent_attr_map, rel_map = {}, {}, {}
    triples = []
    for dataset in datasets:
        print(dataset.name)
        if 'ent_ids' in dataset.name:
            print('entities')
            ent_map = read_entities_map(st.session_state.datapath, dataset.name)
        if 'att_triples' in dataset.name:
            ent_attr_map, ent_attr_set = get_attribute_frequencies(st.session_state.datapath, dataset.name)
        if 'rel_ids' in dataset.name:
            print('relations')
            rel_map = read_entities_map(st.session_state.datapath, dataset.name)
        if 'triples' in dataset.name:
            print('triples')
            triples = get_triples(st.session_state.datapath, dataset.name)

    if len(triples) > 0 and len(ent_map) > 0 and len(rel_map) > 0:
        G = get_KG(triples, ent_map, rel_map)
        return G
    else:
        return {}

def page_generate():
    if 'G1' not in st.session_state or 'G2' not in st.session_state:
        st.markdown("## No graphs have been loaded")
        return 
    emb_dim = st.selectbox('Select dimensionality of node embeddings', [50, 100, 200, 300])
    nr_walks = st.selectbox('Select number of random walks per node', [5, 10, 15, 20])
    walk_length = st.selectbox('Select random walk length', [4, 5, 6, 7, 8, 9, 10])
    generate_button = st.button('Generate training corpus', disabled=False)
    if generate_button:
        st.text('Generate corpora')
        corpus1 = get_corpus(st.session_state.G1, nr_walks=nr_walks, walk_length=walk_length)
        corpus2 = get_corpus(st.session_state.G2, nr_walks=nr_walks, walk_length=walk_length)

        st.text('Computing LLM embeddings')
        llm = get_llm()
        initial_embeddings1 = get_LLM_embeddings(llm, corpus1, emb_dim=emb_dim)
        initial_embeddings2 = get_LLM_embeddings(llm, corpus2, emb_dim=emb_dim)

        write_to_w2v_format(st.session_state.embeddings_path, f'{st.session_state.prefixes[1]}_llm_embeddings.txt', initial_embeddings1)
        write_to_w2v_format(st.session_state.embeddings_path, f'{st.session_state.prefixes[2]}_llm_embeddings.txt', initial_embeddings2)

        print('DeepWalk training')
        
        train_word2vec(corpus1, emb_dim, st.session_state.embeddings_path, \
                       f'{st.session_state.prefixes[1]}_llm_embeddings.txt', f"{st.session_state.prefixes[1]}_deepwalk_embs.txt")
        train_word2vec(corpus2, emb_dim, st.session_state.embeddings_path, \
                       f'{st.session_state.prefixes[2]}_llm_embeddings.txt', f"{st.session_state.prefixes[2]}_deepwalk_embs.txt")

        print('Aggregate neighbor embeddings')
        embs1 = load_embeddings(st.session_state.embeddings_path, f"{st.session_state.prefixes[1]}_deepwalk_embs.txt")
        embs2 = load_embeddings(st.session_state.embeddings_path, f"{st.session_state.prefixes[2]}_deepwalk_embs.txt")

        poly_sketch = None #PolynomialCountSketch(degree=2, n_components=bins, random_state=1) # use the same random state for different graphs
        entity_embs1 = get_vectors(st.session_state.G1, embs1, st.session_state.ent_attr_map1, st.session_state.attr_indices, poly_sketch, bins=emb_dim)
        entity_embs2 = get_vectors(st.session_state.G2, embs2, st.session_state.ent_attr_map2, st.session_state.attr_indices, poly_sketch, bins=emb_dim)

        write_entity_embs_to_file(entity_embs1, st.session_state.embeddings_path, f'{st.session_state.prefixes[1]}_final_embs_{emb_dim}.txt')
        write_entity_embs_to_file(entity_embs2, st.session_state.embeddings_path, f'{st.session_state.prefixes[2]}_final_embs_{emb_dim}.txt')

        # ent_attr_map1, ent_attr_set1 = get_attribute_frequencies(st.session_state.datapath, f'{st.session_state.prefixes[1]}_att_triples')
        # ent_attr_map2, ent_attr_set2 = get_attribute_frequencies(st.session_state.datapath, f'{st.session_state.prefixes[2]}_att_triples')
        # attr_set = ent_attr_set1.intersection(ent_attr_set2)

        # attr_indices = {attr:i for i, attr in enumerate(sorted(list(attr_set)))}

    
def page_train():

    emb_dim = st.selectbox('Select dimensionality of node embeddings', [50, 100, 200, 300])
    train_button = st.button('Train model', disabled=False)
    if train_button:
        entity_embs1 = load_embeddings(st.session_state.embeddings_path, f'{st.session_state.prefixes[1]}_final_embs_{emb_dim}.txt')
        entity_embs2 = load_embeddings(st.session_state.embeddings_path, f'{st.session_state.prefixes[2]}_final_embs_{emb_dim}.txt')

        datapath = f'../data/{st.session_state.prefixes[1]}_{st.session_state.prefixes[2]}/'
        sup_pairs = load_labels(datapath, 'sup_pairs')

        ent_map1 = read_entities_map(datapath, 'ent_ids_1')
        ent_map2 = read_entities_map(datapath, 'ent_ids_2')

        X, y = pairs_to_X_y(sup_pairs, ent_map1, ent_map2,  entity_embs1, entity_embs2, nr_neg=10)
        print('X shape', X.shape)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=73)

        gbm = lgb_model(X_train, X_val, y_train, y_val, nr_rounds=1000)

        gbm.save_model(f"../models/{st.session_state.prefixes[1]}_{st.session_state.prefixes[2]}_model_{emb_dim}.txt")



page_names_to_funcs = {
    "Main Page": page0_main,
    "Feature generation": page_generate,
    "Train model": page_train,
    "Query model": None
} 

def clicked(idx):
    st.session_state.clicked['button'+str(idx)] = True

def process_data(process_button, idx):
    assert idx == 1 or idx == 2
    name = 'button'+str(idx)
    if process_button:
        st.session_state.clicked[name] = True
            
    if st.session_state.clicked[name]:
        uploaded_datasets = st.file_uploader("Upload graph data", accept_multiple_files=True, label_visibility="hidden", key=idx)
        print('# datasets', uploaded_datasets)
        if uploaded_datasets:
            print('generate graph')
            st.session_state['G'+str(idx)] = load_graph(uploaded_datasets)



def main():
    st.set_page_config(page_title="SEAL demo",
                       page_icon = "images/seal2.png")
    #st.write(css, unsafe_allow_html=True)
    st.session_state.disabled =True 

    #st.session_state.datapath = '../data/'
    st.session_state.modelpath = '../models/'
    st.session_state.embeddings_path = '../embeddings/'

    if 'clicked' not in st.session_state:
        st.session_state.clicked = {'button1': False, 'button2': False}

    with st.sidebar:
        st.sidebar.image("../images/seal2.png")
        st.markdown('## SEAL: Scalable Entity ALignment')
        selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]() 

    languages = st.radio('Select pair of graphs', ['French-English', 'Japanese-English', 'Chinese-English'])
    if languages == 'French-English':
        st.session_state.datapath = '../data/fr_en/'
        st.session_state.prefixes = ['dummy', 'fr', 'en']


    st.session_state.ent_map1 = read_entities_map(st.session_state.datapath, 'ent_ids_1')
    st.session_state.ent_map2 = read_entities_map(st.session_state.datapath, 'ent_ids_2')

    st.session_state.ent_attr_map1, ent_attr_set1 = get_attribute_frequencies(st.session_state.datapath, f'{st.session_state.prefixes[1]}_att_triples')
    st.session_state.ent_attr_map2, ent_attr_set2 = get_attribute_frequencies(st.session_state.datapath, f'{st.session_state.prefixes[2]}_att_triples')
    st.session_state.attr_set = ent_attr_set1.intersection(ent_attr_set2)

    st.session_state.attr_indices = {attr:i for i, attr in enumerate(sorted(list(st.session_state.attr_set)))}
    
    st.session_state.rel_map1 = read_entities_map(st.session_state.datapath, 'rel_ids_1')
    st.session_state.rel_map2 = read_entities_map(st.session_state.datapath, 'rel_ids_2')

    st.session_state.G1 = get_graph(st.session_state.datapath, 'triples_1', st.session_state.ent_map1, st.session_state.rel_map1)
    st.session_state.G2 = get_graph(st.session_state.datapath, 'triples_2', st.session_state.ent_map2, st.session_state.rel_map2)
 


if __name__ == '__main__':
    main()