
import re
import numpy as np
from nltk.util import skipgrams

def load_embeddings(datapath, filename):
    f = open(datapath+filename, 'r', encoding='utf8')
    emb_map = {}
    for line in f:
        ls = line.split()
        vals = [float(val) for val in ls[1:]]
        if len(vals) == 1:
            continue
        emb_map[ls[0]] = vals
    f.close()
    return emb_map

def load_labels(datapath, fname):
    f = open(datapath+fname, 'r', encoding='utf8')
    pairs = []
    for line in f:
        ls = line.split()
        pairs.append((ls[0], ls[1]))
    return pairs

def read_entities_map(datapath, filename):
    uri_map = {}
    f = open(datapath+filename, 'r', encoding="utf8")
    for line in f:
        nr_id, uri = line.split()[0], line.split()[1]
        uri_map[nr_id] = uri
    f.close()
    return uri_map

def get_triples(datapath, triplesfile):
    triples = []
    f = open(datapath+triplesfile, 'r')
    for line in f:
        ls = line.split()
        h, r, t = ls[0], ls[1], ls[2]
        triples.append((h,r,t))
    f.close()
    return triples

def get_KG(triples, ent_map, rel_map):
    '''creating a KG from a list of triples
        :params:
        triples: list of (head, relation, tail) triples, represented by integer ids
        ent_map: hash map integer_id to URI for the head and tail entities
        rel_map: hash_map integer_id to URI for the relations
    '''
    graph = {}
    for triple in triples:
        h, r, t = ent_map[triple[0]].split('/')[-1], rel_map[triple[1]].split('/')[-1], ent_map[triple[2]].split('/')[-1]
        graph.setdefault(h, [])
        graph[h].append((r, t))
        graph.setdefault(t, [])
        graph[t].append(('reverse_'+r, h))
    return graph

def split_entity(s, chars_to_remove=['_', '.', ',', '(', ')', '[', ']', '!']):
    '''
    splitting URIs by capital letter and removing special characters 
    '''
    t = re.sub( r"([A-Z]|_)", r" \1", s).split()
    res = ' '.join(t)
    sc = set(chars_to_remove)
    return ''.join([c for c in res if c not in sc])

def get_graph(datapath, triplesfile, ent_map, rel_map):
    graph = {}
    f = open(datapath+triplesfile, 'r')
    for line in f:
        ls = line.split()
        h, r, t = ent_map[ls[0]].split('/')[-1], rel_map[ls[1]].split('/')[-1], ent_map[ls[2]].split('/')[-1]
        graph.setdefault(h, [])
        graph[h].append((r, t))
        graph.setdefault(t, [])
        graph[t].append(('reverse_'+r, h))
    f.close()
    return graph

def write_to_w2v_format(path, fname, embeddings_map):
    '''
        embeddings are written to the specific word2vec format, used for initializing the DeepWalk embeddings 
    '''
    f = open(path + fname, 'w', encoding="utf8")
    f.write(str(len(embeddings_map)) + ' ' + str(len(list(embeddings_map.values())[0])) + '\n')
    for ent, emb in embeddings_map.items():
        f.write(ent + ' ')
        for val in emb:
            f.write(str(val) + ' ')
        f.write('\n')
    f.close()

def write_entity_embs_to_file(embeddings_map, path, fname):
    '''
        writing trained embeddings to a text file in the format name:embedding_vector
        This can be improved by writing the entity names and embeddings to separate files, e.g. npz files for the , as all embeddings have the same length
    '''
    f = open(path + fname, 'w', encoding="utf8")
    for ent, emb in embeddings_map.items():
        f.write(ent + ' ')
        for val in emb:
            f.write(str(val) + ' ')
        f.write('\n')
    f.close()

def get_attribute_frequencies(datapath, filename):
    '''
        get the frequencies of the different attribute types for each entity
    '''
    entity_types_map = {} 
    all_types = set()
    f = open(datapath+filename, 'r', encoding='utf8')
    for line in f:
        ls = line.split('^^')
        entity = ls[0].split()[0]
        entity = entity.split('/')[-1][:-1]
        entity_types_map.setdefault(entity, {'string': 0})
        if len(ls) > 1:
            tp = ls[-1][1:-4]
            tp = tp.split('#')[-1]
            tp = tp.split('/')[-1]
            entity_types_map[entity].setdefault(tp, 0)
            entity_types_map[entity][tp] += 1
            all_types.add(tp)
        else:
            entity_types_map[entity]['string'] += 1
    all_types.add('string')
    f.close()
    return entity_types_map, all_types

def string2vec(string, bins, nr_chars, distance):
    '''
        generate skipgrams with nr_chars characters, see https://tedboy.github.io/nlps/generated/generated/nltk.skipgrams.html
        As the numbers of skip-grams can become very large, we hash them to a fixed number of bins 
    '''
    vec = [0 for _ in range(bins)]
    for sg in list(skipgrams(string, nr_chars, distance)):
        vec[hash(''.join(sg))%bins] += 1
    return vec

def attr2vec(attr_map, attr_indices):
    '''
        for the attribute map of each entity,
        convert the counts to a (sparse) vector
    '''
    vec = [-1 for _ in range(len(attr_indices)+1)] # the missing value can be 0 instead of -1
    for attr, cnt in attr_map.items():
        vec[attr_indices.get(attr, len(vec)-1)] = cnt
    return vec

# def get_vectors(G, embs, ent_attrs, attr_indices, kernel_map, bins, nr_chars=3, distance=1):
#     """
#         G: knowledge graph as a hash map {head: [(rel, tail), ...]}
#         embs: precomputed entity and relations embeddings
#         kernel_map: explicit feature map for non-linear kernels, see https://scikit-learn.org/1.5/modules/kernel_approximation.html#kernel-approximation
#         bins: the dimensionality of the skip-gram frequency vectors
#         nr_chars, distance: skipgram parameters
#     """
#     X =  np.array(list(embs.values()))
#     print(X.shape)
#     #sketch = kernel_map.fit_transform(X)
#     #print('sketch shape', sketch.shape)
#     #assert len(embs) == sketch.shape[0]
#     kernel_embs = embs #{entity: kemb for entity, kemb in zip(embs, sketch)}
#     cnt = 0
#     nf = 0
#     entity_kernel_embs = {}
#     cnt_attrs = 0
#     for node, neighbors in G.items():
#         if len(neighbors) == 0:
#             print(node)
#             continue
#         # print(node)
#         cnt += 1
#         if cnt % 5000 == 0:
#             print(cnt)
#         emb_head = kernel_embs[node]
#         skipgram_head = string2vec(node, bins, nr_chars, distance)
#         #nbr_rels, nbr_tails = [], []
#         nbrs = []
#         skipgram_rels, skipgram_tails = [], []
#         degrees = []
#         node_attr_vec = [-1 for _ in range(len(attr_indices)+1)]
#         if node in ent_attrs:
#             node_attrs = ent_attrs[node]
#             node_attr_vec = attr2vec(node_attrs, attr_indices)
#             cnt_attrs += 1
#         for nbr in neighbors:
#             if nbr[0] not in embs or nbr[1] not in embs:
#                 #print('not found', nf, nbr)
#                 #nf += 1
#                 continue
#             emb_rel, emb_tail = np.array(kernel_embs[nbr[0]]), np.array(kernel_embs[nbr[1]])
#             emb_rel = emb_rel/np.linalg.norm(emb_rel)
#             nbrs.append(np.multiply(emb_rel, emb_tail))
#             #nbr_rels.append(emb_rel)
#             #nbr_tails.append(emb_tail)
#             degrees.append(len(G[nbr[1]]))
#             skipgram_rels.append(string2vec(nbr[0], bins, nr_chars, distance))
#             skipgram_tails.append(string2vec(nbr[1], bins, nr_chars, distance))
#         #emb_rel_agg = np.mean(nbr_rels, axis=0)
#         #emb_tails_agg = np.mean(nbr_tails, axis=0)
#         if len(nbrs) == 0:
#             nf += 1
#             print('nf', nf)
#             continue
#         emb_nbrs = np.sum(nbrs, axis=0)
#         skipgram_rel_agg = np.mean(skipgram_rels, axis=0)
#         skipgram_tails_agg = np.mean(skipgram_tails, axis=0)
#         v = list(emb_head) + list(emb_nbrs) + list(skipgram_head) #+ list(skipgram_rel_agg) + list(skipgram_tails_agg) + node_attr_vec
#         # v = list(emb_head) + list(emb_rel_agg) + list(emb_tails_agg) + list(skipgram_head) + list(skipgram_rel_agg) + list(skipgram_tails_agg)
#         # if len(neighbors) < 1:
#         #     print(node, neighbors)
#         hist, _ = np.histogram(degrees, bins=[0, 3, 6, 10, 15, 20])
#         v.extend(hist)
#         v.append(len(neighbors))
#         entity_kernel_embs[node] = np.array(v)/np.linalg.norm(v)
#     print('#nodes with attributes/not found', cnt_attrs, nf)
#     return entity_kernel_embs


