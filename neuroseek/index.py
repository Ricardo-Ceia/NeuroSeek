from neuroseek.vector import Vector


class Index:
    def __init__(self):
        self.vectors = []
        self.id_to_index = {}
        self._next_id = 0

    def __len__(self):
        return len(self.vectors)

    def get_vector(self, id):
        if not isinstance(id, int):
            raise TypeError(f"unsupported operand type(s) for get_vector: 'Index' and '{type(id).__name__}'")

        if id not in self.id_to_index:
            raise ValueError(f"ID {id} does not exist in index")

        index = self.id_to_index[id]
        return self.vectors[index][1]

    def add_vector(self, vector, id=None):
        if not isinstance(vector, Vector):
            raise TypeError(f"unsupported operand type(s) for add_vector: 'Index' and '{type(vector).__name__}'")

        if id is None:
            id = self._next_id
            while id in self.id_to_index:
                id = self._next_id
                self._next_id += 1

        if not isinstance(id, int):
            raise TypeError(f"unsupported operand type(s) for id: 'Index' and '{type(id).__name__}'")

        if id in self.id_to_index:
            raise ValueError(f"ID {id} already exists. Use update_vector() to replace.")

        index = len(self.vectors)
        self.vectors.append((id, vector))
        self.id_to_index[id] = index
        return id
    
    def delete_vector(self, id=None):
        if id is None:
            raise ValueError("ID must be provided for deletion")

        if not isinstance(id, int):
            raise TypeError(f"unsupported operand type(s) for delete_vector: 'Index' and '{type(id).__name__}'")

        if id not in self.id_to_index:
            raise ValueError(f"ID {id} does not exist in index")

        index = self.id_to_index[id]
        deleted_vector = self.vectors[index]
        del self.vectors[index]
        del self.id_to_index[id]

        for i in range(index, len(self.vectors)):
            self.id_to_index[self.vectors[i][0]] = i

        return deleted_vector

    def update_vector(self, id, vector):
        if not isinstance(id, int):
            raise TypeError(f"unsupported operand type(s) for update_vector: 'Index' and '{type(id).__name__}'")

        if not isinstance(vector, Vector):
            raise TypeError(f"unsupported operand type(s) for update_vector: 'Index' and '{type(vector).__name__}'")

        if id not in self.id_to_index:
            raise ValueError(f"ID {id} does not exist in index")

        index = self.id_to_index[id]
        old_vector = self.vectors[index][1]
        self.vectors[index] = (id, vector)

        return (id, old_vector)
        
    def search(self, query_vector, top_k=5):
        if not isinstance(query_vector, Vector):
            raise TypeError(f"unsupported operand type(s) for search: 'Index' and '{type(query_vector).__name__}'")
        
        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, not {type(top_k).__name__}")
        
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        
        if not self.vectors:
            return []
        
        if len(query_vector) == 0:
            raise ValueError("Cannot search with empty query vector")

        similarities = []
        for id, vector in self.vectors:
            if len(query_vector) != len(vector):
                raise ValueError(f"Query vector dimension {len(query_vector)} does not match stored vector dimension {len(vector)}")
            similarity = query_vector.cosine_similarity(vector)
            similarities.append((id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
