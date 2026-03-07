import pickle
import numpy as np


def save_hnsw_index(index, filename):
    data = {
        'M': index.M,
        'efConstruction': index.efConstruction,
        'maxLayers': index.maxLayers,
        'layers': index.layers,
        'id_to_node': index.id_to_node,
        'entry_point_id': index.entry_point.id if index.entry_point else None,
        'num_vectors': index.num_vectors,
        '_next_id': index._next_id,
        # numpy matrix backing
        '_dim': index._dim,
        '_capacity': index._capacity,
        '_matrix': index._matrix,
        '_id_to_row': index._id_to_row,
        '_row_to_id': index._row_to_id,
        '_free_rows': index._free_rows,
        # reverse adjacency
        '_reverse_adj': index._reverse_adj,
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_hnsw_index(filename, HNSWIndex):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    index = HNSWIndex(
        M=data['M'],
        efConstruction=data['efConstruction'],
        maxLayers=data['maxLayers'],
    )
    index.layers = data['layers']
    index.id_to_node = data['id_to_node']
    index.num_vectors = data['num_vectors']
    index._next_id = data.get('_next_id', index.num_vectors)

    # numpy matrix backing
    index._dim = data.get('_dim', 0)
    index._capacity = data.get('_capacity', 0)
    index._matrix = data.get('_matrix', np.empty((0, 0), dtype=np.float32))
    index._id_to_row = data.get('_id_to_row', {})
    index._row_to_id = data.get('_row_to_id', {})
    index._free_rows = data.get('_free_rows', [])

    # reverse adjacency
    index._reverse_adj = data.get('_reverse_adj', {})

    if data['entry_point_id'] is not None:
        index.entry_point = index.id_to_node[data['entry_point_id']]

    return index
