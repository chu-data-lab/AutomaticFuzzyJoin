from .preprocessor import Preprocessor
from .tokenizer import Tokenizer
from .token_weight import TokenWeight
from .distance_function import DistanceFunction
import pandas as pd
import os
import pickle
from ...utils import makedir


class AutoFJJoinFunction(object):
    """Join function. A join function computes a distance score between a
    pair of records on a given column. Our built-in join function is specified
    by four parameters including preprocess_method, tokenize_method,
    token_weight_method and distance_function.

    Parameters
    ----------
    preprocess_method: string
        Preprocessing method. See preprocessor.py for available methods.

    tokenize_method: string or None
        Tokenization method. See tokenizer.py for available methods.

    token_weight_method: string or None
        Token weighting method. See token_weight.py for available methods.

    distance_function: string
        Distance function. See distance_function.py for available methods.

    """

    def __init__(self,
                 preprocess_method,
                 tokenize_method,
                 token_weight_method,
                 distance_function):
        self.preprocess_method = preprocess_method
        self.tokenize_method = tokenize_method
        self.token_weight_method = token_weight_method
        self.distance_function = distance_function
        self.name = "{}_{}_{}_{}".format(preprocess_method,
                                         tokenize_method,
                                         token_weight_method,
                                         distance_function)

    def compute_distance(self, left, right, LL_blocked, LR_blocked,
                         cache_dir=None):
        """Compute the distance of each tuple pair in the LL and LR blocked table.

        Parameters
        ----------
        left: pd.DataFrame
            A subset of the left table that contains the id column and the
            column to be processed. The id column is named as autofj_id.
            The column to be processed is named as value.

        right: pd.DataFrame
            A subset of the right table that contains the id column and the
            column to be processed. The id column is named as autofj_id.
            The column to be processed is named as value.

        LL_blocked: pd.DataFrame
            The LL blocked table that consists of the id columns and
            the columns to be processed. The id columns are named as
            autofj_id_l and autofj_id_r. The column to be processed is named as
            value_l and value_r.

        LR_blocked: pd.DataFrame
            The LR blocked table that consists of the id columns and
            the columns to be processed. The id columns are named as
            autofj_id_l and autofj_id_r. The column to be processed is named as
            value_l and value_r.

        Returns
        -------
        LL_distance: pd.Series
            Distance of each tuple pair in the LL blocked table.

        LR_distance: pd.Series
            Distance of each tuple pair in the LR blocked table.
        """
        left = left.set_index("autofj_id")
        right = right.set_index("autofj_id")

        # cache paths
        if cache_dir is not None:
            L_pre_path, L_pre_token_path, token_weight_path\
                = self.get_cache_paths(cache_dir, "left")
            R_pre_path, R_pre_token_path, _\
                = self.get_cache_paths(cache_dir, "right")
        else:
            L_pre_path = None
            R_pre_path = None
            L_pre_token_path = None
            R_pre_token_path = None
            token_weight_path = None

        # preprocess
        L_pre = self.preprocess(left["value"], cache_path=L_pre_path)
        R_pre = self.preprocess(right["value"], cache_path=R_pre_path)

        # tokenize
        L_pre_token = self.tokenize(L_pre, cache_path=L_pre_token_path)
        R_pre_token = self.tokenize(R_pre, cache_path=R_pre_token_path)

        # token weights
        token_weights = self.compute_token_weights(L_pre_token,
                                                   cache_path=token_weight_path)

        # apply distance function
        LL_distance = self.apply_distance_function(LL_blocked,
                                                   L_pre_token,
                                                   L_pre_token,
                                                   token_weights)
        LR_distance = self.apply_distance_function(LR_blocked,
                                                   L_pre_token,
                                                   R_pre_token,
                                                   token_weights)
        return LL_distance, LR_distance

    def get_cache_paths(self, cache_dir, table):
        """Get paths of different cache files

        Parameters
        ----------
        cache_dir: string
            Directory to store cache

        table: string
            Either "left" or "right"

        Returns
        -------
        pre_path:
            Path of preprocessing results

        pre_token_path:
            Path of preprocessing and tokenization results

        token_weight_path:
            Path of token weights
        """
        pre_path = makedir([cache_dir, table],
                             "{}.p".format(self.preprocess_method))

        pre_token_path = makedir([cache_dir, table],
                            "{}_{}.p".format(self.preprocess_method,
                                             self.tokenize_method))

        if table == "left":
            token_weight_path = makedir([cache_dir, table],
                                        "{}_{}_{}.p".format(
                                            self.preprocess_method,
                                            self.tokenize_method,
                                            self.token_weight_method))
        else:
            token_weight_path = None
        return pre_path, pre_token_path, token_weight_path

    def preprocess(self, X, cache_path=None):
        """Preprocess records

        Parameters
        ----------
        X: pd.Series
            Input data

        Returns
        -------
        X_pre: pd.Series
            Preprocessed data.
        """
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                X_pre = pickle.load(f)
        else:
            preprocessor = Preprocessor(self.preprocess_method)
            X_pre = preprocessor.preprocess(X)
            self.save_cache(X_pre, cache_path)
        return X_pre

    def tokenize(self, X_pre, cache_path=None):
        """ Tokenize the preprocessed data

        Parameters
        ----------
        X_pre: pd.Series
            Preprocessed data

        Returns
        -------
        X_pre_token: pd.Series
            Data after preprocessing and tokenization
        """
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                X_pre_token = pickle.load(f)
        else:
            tokenizer = Tokenizer(self.tokenize_method)
            X_pre_token = tokenizer.tokenize(X_pre)
            self.save_cache(X_pre_token, cache_path)
        return X_pre_token

    def compute_token_weights(self, X_pre_token, cache_path=None):
        """ Compute token weights

        Parameters
        ----------
        X_pre_token: pd.Series
            Data after preprocessing and tokenization

        Returns
        -------
        token_weights: dict
            Token weights
        """
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                token_weights = pickle.load(f)
        else:
            weight = TokenWeight(self.token_weight_method)
            token_weights = weight.weight(X_pre_token)
            self.save_cache(token_weights, cache_path)
        return token_weights

    def apply_distance_function(self, LR_blocked, L_pre_token, R_pre_token,
                                token_weights):
        """ Apply distance functions on blocked record pairs

        Parameters
        ----------
        LR_blocked: pd.DataFrame
            A subset of the blocked table that contains the id columns and
            the columns to be processed. The id columns are named as
            autofj_id_l and autofj_id_r. The column to be processed is named as
            value_l and value_r.

        L_pre_token: pd.Series
            Left records after preprocessing and tokenization

        R_pre_token: pd.Series
            Right records after preprocessing and tokenization

        token_weights: dict
            Token weights

        Returns
        -------
        distance: pd.Series
            distance between record pairs in LR_blocked
        """
        # apply join function on record pairs in LR_blocked
        L_id = LR_blocked["autofj_id_l"].values
        R_id = LR_blocked["autofj_id_r"].values

        L = L_pre_token.loc[L_id]
        R = R_pre_token.loc[R_id]
        LR = pd.DataFrame({"value_l": L.values, "value_r": R.values})

        df = DistanceFunction(self.distance_function)
        distance = df.compute_distance(LR, token_weights).values
        return distance

    def save_cache(self, data, cache_path):
        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

# if __name__ == '__main__':
#
#     left = pd.read_csv("../../data/left.csv")[["id", "name"]]
#     left = left.rename(columns={"id":"autofj_id", "name":"value"})
#     right = pd.read_csv("../../data/left.csv")[["id", "name"]]
#     right = right.rename(columns={"id":"autofj_id", "name":"value"})
#     LR_blocked = pd.read_csv("../../data/LR_blocked.csv")
#
#     jf = AutoFJJoinFunction("lower", None, None, "editDistance")
#     tic = time.time()
#     distance = jf.compute_distance(left, right, LR_blocked)
#     print(time.time() - tic)
