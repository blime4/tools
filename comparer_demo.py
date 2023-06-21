import os
from hooktools import Comparer
import hooktools

# config_path = os.path.join(hooktools.__path__[0], "config")

if __name__ == "__main__":
    compare = Comparer(("./comparer_demo.yaml"))
    compare.compare()