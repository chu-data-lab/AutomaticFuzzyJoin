import collections
import math
import pandas as pd
import time
import numpy as np

class defaultdict(dict):
    def set_default_value(self, default_value):
        self.default_value = default_value
        
    def __missing__(self, key):
        self[key] = self.default_value
        return self.default_value

def uniformWeight(document):
    """Uniform weight"""
    weight = defaultdict()
    weight.set_default_value(1)
    return weight

def idfWeight(document):
    """Compute idf weight for tokens

    Parameters:
    -----------
    document: list of sets
        A list of token sets, which is the document on which the idf is
        computed.

    Return:
    -------
    weight: dict
        idf weight of tokens
    """
    token_count = collections.defaultdict(set)

    for i, row in enumerate(document):
        for token in row:
            token_count[token].add(i)

    # calculate idf value
    weight = defaultdict()
    weight.set_default_value(math.log(len(document)))

    for k, v in token_count.items():
        weight[k] = math.log(len(document) / (len(v) + 1))
    return weight

class TokenWeight(object):
    """Token weight

    Parameters
    ----------
    method: string
        Token weighting schema. The available methods are listed as follows.
        - uniformWight
        - idfWeight
        - None (no weights)
    """
    def __init__(self, method):
        self.method = method
        if method is None:
            self.func = None
        elif method == "uniformWeight":
            self.func = uniformWeight
        elif method == "idfWeight":
            self.func = idfWeight
        else:
            raise Exception("{} is an invalid weighting schema"
                             .format(method))
    
    def weight(self, X):
        """ Weight tokens

        Parameters
        ----------
        X: pd.Series
            Input data

        Return
        ------
        weight: dict
            weight of tokens
        """
        if self.func is not None:
            weight = self.func(X)
            return weight
        else:
            return None
#
# data = pd.read_csv("../../data/left.csv")["title"]
# X = np.concatenate([data.values for _ in range(20)])
# X = pd.Series(X)
#
# weight = TokenWeight("idfWeight")
# tic = time.time()
# weight.weight(X)
# print(time.time() - tic)
