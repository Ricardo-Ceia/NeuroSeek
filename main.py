class Vector:
    def __init__(self,size):
        self.size = size
        self.data = [0] * size

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return self.size
    
    def __repr__(self):
        return f"{self.data}"
    
    def __add__(self,other):
        if len(self)!=len(other):
            raise ValueError("Vectors must be of the same size to be added.")
        result = Vector(len(self))
        for i in range(len(self)):
            result.data[i] = self.data[i] + other.data[i]
        return result

    def __sub__(self,other):
        if len(self)!=len(other):
            raise ValueError("Vectors must be of the same size to be subtracted.")
        result = Vector(len(self))
        for i in range(len(self)):
            result.data[i] = self.data[i] - other.data[i]
        return result

    def __mul__(self,other):
        if len(self)!=len(other):
            raise ValueError("Vectors must be of the same size to be multiplied.")
        result = Vector(len(self))
        for i in range(len(self)):
            result.data[i] = self.data[i] * other.data[i]
        return result
    

    def __dot__(self,other):
        if len(self)!=len(other):
            raise ValueError("Vectors must be the same size to apply dot product on them.")
        result = 0
        for i in range(len(self)):
            result += self.data[i] * other.data[i]
        return result
    
    def __rmul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"unsupported operand type(s) for *: '{type(scalar).__name__}' and 'Vector'")
        result = Vector(len(self))
        for i in range(len(self)):
            result.data[i] = self.data[i] * scalar
        return result

    def __norm__(self):
        result = 0
        for i in range(len(self)):
            result += self.data[i] ** 2
        return result ** 0.5

    def __eq__(self,other):
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

def main():
    print("Hello,World!");
    testVec = Vector(5)
    

main()
