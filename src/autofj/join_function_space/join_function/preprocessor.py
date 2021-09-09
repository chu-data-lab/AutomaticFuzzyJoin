import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
import time
import pandas as pd
import re

ps = PorterStemmer()
# ps = SnowballStemmer("english")

def lower(x):
    return str(x).lower()

def removePunctuation(x):
    return re.sub(r'[^\w\s]', '', x)

def stem(x):
    return " ".join([ps.stem(w) for w in x.split()])

def lowerStem(x):
    x = lower(x)
    x = stem(x)
    return x

def lowerRemovePunctuation(x):
    x = lower(x)
    x = removePunctuation(x)
    return x

def lowerRemovePunctuationStem(x):
    x = lower(x)
    x = removePunctuation(x)
    x = stem(x)
    return x

class Preprocessor:
    """Preprocess data

    Parameters
    ----------
    method: string
        Preprocessing method. The available methods are listed as follows.
        - lower: lowercase
        - lowerStem: lowercase and stem
        - lowerRemovePunctuation: lowercase and remove punctuation
        - lowerRemovePunctuationStem: lowercase, remove punctuation and stem
    """
    def __init__(self, method):
        self.method = method
        if method == "lower":
            self.func = lower
        elif method == "lowerStem":
            self.func = lowerStem
        elif method == "lowerRemovePunctuation":
            self.func = lowerRemovePunctuation
        elif method == "lowerRemovePunctuationStem":
            self.func = lowerRemovePunctuationStem
        else:
            raise Exception("{} is an invalid preprocessing method"
                             .format(method))
    def preprocess(self, X):
        """ Preprocess the given data

        Parameters
        ----------
        X: pd.Series
            Input data
        """
        X = X.apply(self.func)
        return X

# data = pd.read_csv("../../data/left.csv")["title"]
# X = np.concatenate([data.values for _ in range(20)])
# X = pd.Series(X)

# pre1 = Preprocessor("lowerRemovePunctuationStem")
# tic = time.time()
# pre1.preprocess(X)
# print(time.time() - tic)

# pre2 = OldPreprocess(X)
# tic = time.time()
# pre2.process(("lower", "remove_punctuation", "stem"))
# print(time.time() - tic)

