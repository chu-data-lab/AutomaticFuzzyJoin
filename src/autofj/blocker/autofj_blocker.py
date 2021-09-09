import pandas as pd
import math
import collections
import ngram
import os
from multiprocessing import Pool
import re
from nltk.stem.porter import PorterStemmer
import time
from functools import partial
import numpy as np

class AutoFJBlocker(object):
    """AutoFJ default Blocker

    Parameters
    ----------
    num_candidates: int or None
        Number of candidates for each right record. If None, the number will be
        the square root of the number of records in the left table

    n_jobs : int, default=1
        Number of CPU cores used. -1 means using all processors.
    """
    def __init__(self, num_candidates=None, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.num_candidates = num_candidates

    def block(self, left_table, right_table, id_column):
        """ Perform blocking on two tables

        Parameters
        ----------
        left_table: pd.DataFrame
            Reference table. The left table is assumed to be almost
            duplicate-free, which means it has no or only few duplicates.

        right_table: pd.DataFrame
            Another input table.

        id_column: string
            The name of id column in two tables.

        Returns:
        --------
            result: pd.DataFrame
            A table of records pairs survived blocking. Column names
            id_column + "_l" and id_column + "_r"
        """
        self.id_column = id_column

        # preprocess records
        left = self._preprocess(left_table, id_column)
        right = self._preprocess(right_table, id_column)

        # get num candidates
        if self.num_candidates is None:
            self.num_candidates = min(int(math.pow(len(left), 0.5)), 50)

        # build token maps using the left table
        token_lid_map, token_idf_map, token_tf_map = self._build_token_maps(
            left)

        # get candidates for each right record
        result = self._get_candidates_multi(right, token_lid_map,
                                      token_idf_map, token_tf_map)

        result = result.rename(columns={"id_l": id_column+"_l",
                                        "id_r": id_column + "_r"})
        return result

    def _preprocess(self, df, id_column):
        """ Preprocess the records: (1) concatenate all columns. (2) lowercase,
        remove punctuation and do stemming

        Parameters
        ----------
        df: pd.DataFrame
            Original table

        id_column: string
            The name of id column in two tables.

        Reutrn
        ------
        result: pd.DataFrame
            Preprocessed table that has two columns, am id column named as "id"
            and a column for preprocessed record named "value"
        """
        # get column names except id
        columns = [c for c in df.columns if c != id_column]
        ps = PorterStemmer()

        # concat all columns, lowercase, remove punctuation, split by space,
        # and do stemming
        new_value = []
        for x in df[columns].values:
            concat_x = " ".join([str(i) for i in x])
            lower_x = re.sub('[^\w\s]', " ", concat_x.lower())
            stem_x = [ps.stem(w) for w in lower_x.split()]
            new_x = " ".join(stem_x)
            new_value.append(new_x)

        id_df = df[id_column].values
        result = pd.DataFrame({"id":id_df, "value":new_value})
        return result

    def _build_token_maps(self, left):
        """ Build token maps for blocking

        Parameters:
        -----------
        left: pd.DataFrame
            The left table

        Return:
        token_id_map: dict {token: left_id_set]}
            A dictionary that maps each token to id of left records contain
            the token

        token_idf_map: dict {token: idf_score}
            A dictionary that maps each token to its idf score

        token_tf_map: dict {token: {left_id: tf_score}}
            A dictionary that maps each token to its tf scores in different
            left records.
        """
        token_lid_map = collections.defaultdict(set)
        token_tf_map = collections.defaultdict(dict)
        three_gram = ngram.NGram(N=3)

        # build token maps
        for lid, lvalue in left[["id", "value"]].values:
            if len(lvalue) == 0:
                continue

            # 3-gram tokenization + white space tokenization
            tokens = list(three_gram.split(lvalue)) + list(lvalue.split())

            # add lid to id map
            for token in tokens:
                token_lid_map[token].add(lid)

            #compute tf score
            counter = collections.Counter(tokens)

            for token, count in counter.items():
                token_tf_map[lid][token] = count / len(tokens)

        # compute idf score
        token_idf_map = {}
        for token, lids in token_lid_map.items():
            token_idf_map[token] = math.log(len(left) / (len(lids)+1))
        return token_lid_map, token_idf_map, token_tf_map

    def _get_candidates(self, right, token_lid_map, token_idf_map,
                            token_tf_map):
        """ Get candidates for one record in right table
        Parameters:
        -----------
        right: pd.DataFrame
            Right table

        token_id_map: dict {token: left_id_set]}
            A dictionary that maps each token to id of left records contain
            the token

        token_idf_map: dict {token: idf_score}
            A dictionary that maps each token to its idf score

        token_tf_map: dict {token: {left_id: tf_score}}
            A dictionary that maps each token to its tf scores in different
            left records.

        Return:
        -------
        result: pd.DataFrame
            A table with two columns "id_l", "id_r" that are ids of candidate
            record pairs
        """
        three_gram = ngram.NGram(N=3)
        result = []
        
        for rid, rvalue in right[["id", "value"]].values:
            tokens = set(list(three_gram.split(rvalue)) + list(rvalue.split()))
            counter = collections.defaultdict(int)

            for token in tokens:
                for lid in token_lid_map[token]:
                    counter[lid] += token_idf_map[token] *\
                                    token_tf_map[lid][token]

            if len(counter) <= self.num_candidates:
                candidate_lids = list(counter.keys())
            else:
                counter_sorted = sorted(counter, key=counter.get)[::-1]
                thresh = counter[counter_sorted[self.num_candidates-1]]
                candidate_lids = [lid for lid, value in counter.items()
                                  if value >= thresh]

            for lid in candidate_lids:
                result.append([lid, rid])

        result = pd.DataFrame(result, columns=["id_l", "id_r"])
        return result

    def _get_candidates_multi(self, right, token_lid_map, token_idf_map,
                            token_tf_map):
        """ Get candidates for one record in right table using multiple cpus
        Parameters:
        -----------
        right: pd.DataFrame
            Right table

        token_id_map: dict {token: left_id_set]}
            A dictionary that maps each token to id of left records contain
            the token

        token_idf_map: dict {token: idf_score}
            A dictionary that maps each token to its idf score

        token_tf_map: dict {token: {left_id: tf_score}}
            A dictionary that maps each token to its tf scores in different
            left records.

        Return:
        -------
        result: pd.DataFrame
            A table with two columns "id_l", "id_r" that are ids of candidate
            record pairs
        """
        right_groups = np.array_split(right, self.n_jobs)
        func = partial(self._get_candidates, token_lid_map=token_lid_map,
                       token_idf_map=token_idf_map, token_tf_map=token_tf_map)

        with Pool(self.n_jobs) as pool:
            results = pool.map(func, right_groups)

        results = pd.concat(results, axis=0).sort_values(by=["id_r", "id_l"])
        return  results

if __name__ == '__main__':
    # Test Case
    tic = time.time()
    blocker = AutoFJBlocker(n_jobs=4)
    left = pd.read_csv("data/left.csv")
    right = pd.read_csv("data/right.csv")
    result = blocker.block(left, right, 'id')
    print(result)
    print(time.time() - tic)