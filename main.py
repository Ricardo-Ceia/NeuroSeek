class Vector:
    def __init__(self,size):
        self.size = size
        self.data = [0] * size

def main():
    print("Hello,World!");
    testVec = Vector(5)
    print(testVec.data)

main()
