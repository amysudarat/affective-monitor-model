# -*- coding: utf-8 -*-

"""
contains main loop for training
"""

import model.data_loader
import pandas as pd
import numpy as np


def main():
    # load graph of face
    input_data = model.data_loader.load_facial_graph()


