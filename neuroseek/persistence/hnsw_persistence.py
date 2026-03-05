import pickle


def save_hnsw_index(index, filename):
    data = {
        'M': index.M,
        'efConstruction': index.efConstruction,
        'maxLayers': index.maxLayers,
        'layers': index.layers,
        'id_to_node': index.id_to_node,
        'entry_point_id': index.entry_point.id if index.entry_point else None,
        'num_vectors': index.num_vectors
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_hnsw_index(filename, HNSWIndex):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    index = HNSWIndex(
        M=data['M'],
        efConstruction=data['efConstruction'],
        maxLayers=data['maxLayers']
    )
    index.layers = data['layers']
    index.id_to_node = data['id_to_node']
    index.num_vectors = data['num_vectors']

    if data['entry_point_id'] is not None:
        index.entry_point = index.id_to_node[data['entry_point_id']]

    return index
