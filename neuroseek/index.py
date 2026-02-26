from neuroseek.vector import Vector


class Index:
    def __init__(self):
        self.vectors = []
        self.id_to_index = {}
        self._next_id = 0

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
