import os
from hooktools import Comparer

if __name__ == "__main__":
    compare = Comparer(os.path.join(os.path.dirname(__file__), "comparer_demo.yaml"))
    compare.compare()