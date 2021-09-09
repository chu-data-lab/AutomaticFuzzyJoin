import ngram
import pandas as pd
import numpy as np
import time

three_gramer = ngram.NGram(N=3)

def splitBySpace(x):
    return x.split()

def threeGram(x):
    # Replace whitespace more than one by a single blank
    return list(three_gramer.split(" ".join(x.split())))

class Tokenizer:
    """Tokenize data

    Parameters
    ----------
    method: string
        Tokenization method. The available methods are listed as follows.
        - splitBySpace
        - threeGram
        - None (no tokenization)
    """
    def __init__(self, method):
        self.method = method
        if method is None:
            self.func = None
        elif method == "splitBySpace":
            self.func = splitBySpace
        elif method == "threeGram":
            self.func = threeGram
        else:
            raise Exception("{} is an invalid tokenization method"
                             .format(method))

    def tokenize(self, X):
        """ Tokenize input data

        Parameters
        ----------
        X: pd.Series
            Input data
        """
        if self.func is not None:
            X = X.apply(self.func)
        return X

# data = pd.read_csv("../../data/left.csv")["title"]
# X = np.concatenate([data.values for _ in range(20)])
# X = pd.Series(X)
#
# tokenizer = Tokenizer("threeGram")
# tic = time.time()
# tokenizer.tokenize(X)
# print(time.time() - tic)