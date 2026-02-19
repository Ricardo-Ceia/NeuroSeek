class Vector:
    def __init__(self,size):
        self.size = size
        self.data = [0] * size

    def __getitem_(self,index):
        return self.data[index]

    def __len__(self):
        return self.size
    
    def __repr__(self):
        return f"{self.data}"

def main():
    print("Hello,World!");
    testVec = Vector(5)
    print(len(testVec))

main()
