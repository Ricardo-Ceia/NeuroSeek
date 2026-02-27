import pickle
from neuroseek.vector import Vector


def save_index(index, filename):
    data = {
        'vectors': [(id, vector.data) for id, vector in index.vectors],
        'id_to_index': index.id_to_index,
        '_next_id': index._next_id
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_index(index, filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    vectors = []
    for id, vector_data in data['vectors']:
        vector = Vector(len(vector_data))
        vector.data = vector_data
        vectors.append((id, vector))

    index.vectors = vectors
    index.id_to_index = data['id_to_index']
    index._next_id = data['_next_id']

    return index
