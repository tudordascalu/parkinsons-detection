"""
Scope:
   Data loading and processing
"""

import numpy as np
import pandas as pd


def load_data(file="parkisons_test"):
    """
    Arguments:
        file
    Returns:
        X, y from file
    """
    data = np.genfromtxt("data/" + file)
    return data[:, :-1], data[:, -1]
