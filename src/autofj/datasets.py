import pandas as pd
import os
from os.path import dirname

def load_data(name):
    module_path = dirname(__file__)
    if os.path.exists(os.path.join(module_path, "benchmark", name)):
        left_table = pd.read_csv(os.path.join(module_path, "benchmark", name, "left.csv"))
        right_table = pd.read_csv(os.path.join(module_path, "benchmark", name, "right.csv"))
        gt_table = pd.read_csv(os.path.join(module_path, "benchmark", name, "gt.csv"))
        return left_table, right_table, gt_table
    else:
        raise Exception("Dataset {} does not exist.".format(name))
