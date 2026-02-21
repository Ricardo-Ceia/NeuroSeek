class Vector:
    def __init__(self,size):
        self.size = size
        self.data = [0] * size

    def __getitem__(self,index):
        if not isinstance(index, int):
            raise TypeError(f"indices must be integers, not {type(index).__name__}")
        if index < -len(self) or index >= len(self):
            raise IndexError(f"index {index} out of range for vector of size {len(self)}")
        return self.data[index]
    
    def __setitem__(self,index,value):
        if not isinstance(index, int):
            raise TypeError(f"indices must be integers, not {type(index).__name__}")
        if index < -len(self) or index >= len(self):
            raise IndexError(f"index {index} out of range for vector of size {len(self)}")
        self.data[index] = value

    def __len__(self):
        return self.size
    
    def __repr__(self):
        return f"{self.data}"
    
    def __add__(self,other):
        if not isinstance(other, Vector):
            raise TypeError(f"unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'")
        if len(self)!=len(other):
            raise ValueError("Vectors must be of the same size to be added.")
        result = Vector(len(self))
        for i in range(len(self)):
            result.data[i] = self.data[i] + other.data[i]
        return result

    def __sub__(self,other):
        if not isinstance(other, Vector):
            raise TypeError(f"unsupported operand type(s) for -: 'Vector' and '{type(other).__name__}'")
        if len(self)!=len(other):
            raise ValueError("Vectors must be of the same size to be subtracted.")
        result = Vector(len(self))
        for i in range(len(self)):
            result.data[i] = self.data[i] - other.data[i]
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = Vector(len(self))
            for i in range(len(self)):
                result.data[i] = self.data[i] * other
            return result
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be of the same size to be multiplied.")
            result = Vector(len(self))
            for i in range(len(self)):
                result.data[i] = self.data[i] * other.data[i]
            return result
        else:
            raise TypeError(f"unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")
    

    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError(f"unsupported operand type(s) for dot product: 'Vector' and '{type(other).__name__}'")
        if len(self) != len(other):
            raise ValueError("Vectors must be the same size to apply dot product on them.")
        result = 0
        for i in range(len(self)):
            result += self.data[i] * other.data[i]
        return result

    def __matmul__(self, other):
        return self.dot(other)
    
    def norm(self):
        result = 0
        for i in range(len(self)):
            result += self.data[i] ** 2
        return result ** 0.5

    def __eq__(self,other):
        if not isinstance(other, Vector):
            raise TypeError(f"unsupported operand type(s) for ==: 'Vector' and '{type(other).__name__}'")
        if len(self)!=len(other):
            return False
        for i in range(len(self)):
            if self.data[i]!=other.data[i]:
                return False
        return True

    def __neg__(self):
        result=  Vector(len(self))
        for i in range(len(self)):
            result.data[i] = -self.data[i]
        return result
    
    def __iter__(self):
        return iter(self.data)

    def __contains__(self,item):
        if not isinstance(item, (int, float)):
            raise TypeError(f"unsupported operand type(s) for in: '{type(item).__name__}' and 'Vector'")
        return item in self.data
    
    def cosine_similarity(self,other):
    #(v1, v2) = (v1 · v2) / (||v1|| * ||v2||)
        if not isinstance(other,Vector):
            raise TypeError(f"unsupported operand type(s) for cosine_similarity: 'Vector' and '{type(other).__name__}'")
        if len(self)!=len(other):
            raise ValueError("Vectors must be the same size to apply cosine similarity on them.")
        dot_product = self.dot(other)
        norm_self = self.norm()
        norm_other = other.norm()
        if norm_self == 0 or norm_other == 0: #x/0 tends to infinity 
            raise ValueError("Cosine similarity is not defined for zero-length vectors.")
        return dot_product / (norm_self * norm_other)

def main():
    print("Hello,World!");
    testVec = Vector(5)
    

main()
