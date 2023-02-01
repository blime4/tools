from contextlib import contextmanager
import io
import os
import copy

import time
import json
import hashlib


class Comparer(object):
    
    def __init__(self):
        print("Comparer")
