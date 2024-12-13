

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
    f = open(path + fname, 'w', encoding="utf8")
    f.write(str(len(embeddings_map)) + ' ' + str(len(list(embeddings_map.values())[0])) + '\n')
    for ent, emb in embeddings_map.items():
        f.write(ent + ' ')
        for val in emb:
            f.write(str(val) + ' ')
        f.write('\n')
    f.close()

def write_entity_embs_to_file(embeddings_map, path, fname):
    f = open(path + fname, 'w', encoding="utf8")
    for ent, emb in embeddings_map.items():
        f.write(ent + ' ')
        for val in emb:
            f.write(str(val) + ' ')
        f.write('\n')
    f.close()

def get_attribute_frequencies(datapath, filename):
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
