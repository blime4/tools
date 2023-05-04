import os
import hooktools
from hooktools import Comparer

config_path = os.path.join(hooktools.__path__[0], "config")

if __name__ == "__main__":
    compare = Comparer(os.path.join(config_path, "comparer_demo.yaml"))
    compare.compare()