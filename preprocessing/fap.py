# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def median_filter(fap_signal):
    output = np.median(fap_signal)
    return output