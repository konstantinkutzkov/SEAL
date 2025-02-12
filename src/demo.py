import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import time
from utils import *
from entity_embeddings import *
from train_model import *
import os



def get_llm(llm_type):
    if llm_type == 'tf':
        model_name="https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual/2" 
        llm_model = hub.load(model_name)
    else:
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
        llm_model = SentenceTransformer(model_name)
    return llm_model


def load_graph(datasets, printit=False):
    '''
        Loading datasets into a KG representation.
        The function is specific for the provided input data format, alternative functions should be used for other inputs.
    '''
    ent_map, ent_attr_map, rel_map = {}, {}, {}
    triples = []
    for dataset in datasets:
        print(dataset.name)
        if 'ent_ids' in dataset.name:
            if printit:
                print('entities')
            ent_map = read_entities_map(st.session_state.datapath, dataset.name)
        if 'att_triples' in dataset.name:
            if printit:
                print('Attributes')
            ent_attr_map, ent_attr_set = get_attribute_frequencies(st.session_state.datapath, dataset.name)
        if 'rel_ids' in dataset.name:
            if printit:
                print('relations')
            rel_map = read_entities_map(st.session_state.datapath, dataset.name)
        if 'triples' in dataset.name:
            if printit:
                print('triples')
            triples = get_triples(st.session_state.datapath, dataset.name)

    if len(triples) > 0 and len(ent_map) > 0 and len(rel_map) > 0:
        G = get_KG(triples, ent_map, rel_map)
        return G
    else:
        return {}
    
def page_main():
    st.markdown("#### Scalable Entity ALignment (SEAL) demo")
    languages = st.radio('Select pair of graphs', ['French-English', 'Japanese-English', 'Chinese-English'])
    if languages == 'French-English':
        st.session_state.datapath = '../data/fr_en/'
        st.session_state.prefixes = ['dummy', 'fr', 'en']
    else:
        st.markdown('### :red[Currently not implemented]')


def page_generate():
    if 'G1' not in st.session_state or 'G2' not in st.session_state:
        st.markdown("## No graphs have been loaded")
        return 
    emb_dim = st.selectbox('Select dimensionality of node embeddings', [50, 100, 200, 300])
    nr_walks = st.selectbox('Select number of random walks per node', [5, 10, 15, 20])
    walk_length = st.selectbox('Select random walk length', [4, 5, 6, 7, 8, 9, 10])
    llm_type = st.selectbox('Select LLM for initial entity embeddings', ['TensorFlow Universal-Sentence-Encoder', 'Multilingual-MiniLM-L12'])
    if 'TensorFlow' in llm_type:
        llm_type = 'tf'
    else:
        llm_type = 'minilm'
    fname1 = f'{st.session_state.prefixes[1]}_final_embs_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt'
    fname2 = f'{st.session_state.prefixes[2]}_final_embs_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt'
    if os.path.exists(st.session_state.embeddings_path + fname1) and os.path.exists(st.session_state.embeddings_path + fname2):
            st.markdown('###### :red[Embeddings for the given hyperparameters already exist. Do you want to compute them again?]')
    generate_button = st.button('Generate training corpus', disabled=False)
    if generate_button:
        start = time.time()
        st.text('Generate corpora')
        # generating a corpus with random walks ent1-rel1-ent2-rel2-ent3-.... 
        corpus1 = get_corpus(st.session_state.G1, nr_walks=nr_walks, walk_length=walk_length)
        corpus2 = get_corpus(st.session_state.G2, nr_walks=nr_walks, walk_length=walk_length)

        st.text('Computing LLM embeddings')
        llm_model = get_llm(llm_type)
        initial_embeddings1 = get_LLM_embeddings(llm_model, llm_type, corpus1, emb_dim=emb_dim)
        initial_embeddings2 = get_LLM_embeddings(llm_model, llm_type, corpus2, emb_dim=emb_dim)

        write_to_w2v_format(st.session_state.embeddings_path, f'{st.session_state.prefixes[1]}_llm_embeddings.txt', initial_embeddings1)
        write_to_w2v_format(st.session_state.embeddings_path, f'{st.session_state.prefixes[2]}_llm_embeddings.txt', initial_embeddings2)

        st.text('DeepWalk training')
        
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

        write_entity_embs_to_file(entity_embs1, st.session_state.embeddings_path, \
                                  f'{st.session_state.prefixes[1]}_final_embs_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt')
        write_entity_embs_to_file(entity_embs2, st.session_state.embeddings_path, \
                                  f'{st.session_state.prefixes[2]}_final_embs_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt')
        st.markdown(f'##### Feature generation finished in {np.round(time.time()-start, 2)} seconds.')

    
def page_train():
    emb_dim = st.selectbox('Select dimensionality of node embeddings', [50, 100, 200, 300])
    nr_walks = st.selectbox('Select number of random walks per node', [5, 10, 15, 20])
    walk_length = st.selectbox('Select random walk length', [4, 5, 6, 7, 8, 9, 10])
    llm_type = st.selectbox('Select LLM for initial entity embeddings', ['TensorFlow Universal-Sentence-Encoder', 'Multilingual-MiniLM-L12'])
    if 'TensorFlow' in llm_type:
        llm_type = 'tf'
    else:
        llm_type = 'minilm'
    train_button = st.button('Train model', disabled=False)
    if train_button:
        fname1 = f'{st.session_state.prefixes[1]}_final_embs_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt'
        fname2 = f'{st.session_state.prefixes[2]}_final_embs_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt'
        if not os.path.exists(st.session_state.embeddings_path + fname1) or not os.path.exists(st.session_state.embeddings_path + fname2):
            st.markdown('### No embeddings for the given hyperparameters.')
            return 
        entity_embs1 = load_embeddings(st.session_state.embeddings_path, fname1)
        entity_embs2 = load_embeddings(st.session_state.embeddings_path, fname2)

        datapath = f'../data/{st.session_state.prefixes[1]}_{st.session_state.prefixes[2]}/'
        sup_pairs = load_labels(datapath, 'sup_pairs')

        ent_map1 = read_entities_map(datapath, 'ent_ids_1')
        ent_map2 = read_entities_map(datapath, 'ent_ids_2')

        X, y = pairs_to_X_y(sup_pairs, ent_map1, ent_map2,  entity_embs1, entity_embs2, nr_neg=10)
        print('X shape', X.shape)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=73)

        gbm = lgb_model(X_train, X_val, y_train, y_val, nr_rounds=3000)
        gbm.save_model(f"../models/{st.session_state.prefixes[1]}_{st.session_state.prefixes[2]}_model_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt")
        st.text('Training done')


def page_query():
    emb_dim = st.selectbox('Select dimensionality of node embeddings', [50, 100, 200, 300])
    nr_walks = st.selectbox('Select number of random walks per node', [5, 10, 15, 20])
    walk_length = st.selectbox('Select random walk length', [4, 5, 6, 7, 8, 9, 10])
    llm_type = st.selectbox('Select LLM for initial entity embeddings', ['TensorFlow Universal-Sentence-Encoder', 'Multilingual-MiniLM-L12'])
    if 'TensorFlow' in llm_type:
        llm_type = 'tf'
    else:
        llm_type = 'minilm'
    fname1 = f'{st.session_state.prefixes[1]}_final_embs_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt'
    fname2 = f'{st.session_state.prefixes[2]}_final_embs_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt' 
    if not os.path.exists(st.session_state.embeddings_path + fname1) or not os.path.exists(st.session_state.embeddings_path + fname2):
        st.markdown('#### No pretrained embeddings for the given hyperparameters.')
        return 
    entity_embs1 = load_embeddings(st.session_state.embeddings_path, fname1)
    entity_embs2 = load_embeddings(st.session_state.embeddings_path, fname2)
    model_file = f"../models/{st.session_state.prefixes[1]}_{st.session_state.prefixes[2]}_model_{emb_dim}_{nr_walks}_{walk_length}_{llm_type}.txt"
    if not os.path.exists(model_file):
        st.markdown('#### No pretrained model for the given hyperparameters.')
        return 
    lgb_model = lgb.Booster(model_file=model_file)
    ref_pairs = load_labels(st.session_state.datapath, 'ref_pairs')
    queries = {}
    for pair in ref_pairs[:1000]:
        name = st.session_state.ent_map1[pair[0]].split('/')[-1]
        queries[name] = st.session_state.ent_map2[pair[1]].split('/')[-1]
    inp = st.selectbox('Select query', list(queries.keys()))
    q = queries[inp]
    feats, entities = [], []
    for ent, emb in entity_embs2.items():
        feats.append(list(entity_embs1[inp]) + list(emb))
        entities.append(ent)
    df = pd.DataFrame(feats)
    preds = lgb_model.predict(df)
    for i, index in enumerate(preds.argsort()[-20:][::-1]):
        if entities[index] == q:
            st.write(f"{i+1}: :green[{entities[index]}]")
        else:
            st.write(str(i+1) + ': ' + entities[index])
    

page_names_to_funcs = {
    "Main Page": page_main,
    "Feature generation": page_generate,
    "Train model": page_train,
    "Query model": page_query
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