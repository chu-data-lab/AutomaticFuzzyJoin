import pandas as pd
from ..utils import print_log
from multiprocessing import Pool
from functools import partial
from .join_function.autofj_join_function import AutoFJJoinFunction
from .options import autofj_lg, autofj_md, autofj_sm
import os
import shutil

class AutoFJJoinFunctionSpace(object):
    """AutoFJ Configuration Space. The space is specified by the space of join
    functions, the space of distance thresholds and the space of column weights.

    Parameters
    ----------
    join_function_space: string or dict or list, default="autofj_sm"
        There are following three ways to define the space of join functions:
        (1) Use the name of built-in join function space. There are three
        options, including "autofj_lg", "autofj_lg" and "autofj_sm" that use
        136, 68 and 14 join functions, respectively. Using less join functions
        can improve efficiency but may worsen performance.
        (2) Use a dict specifying the options for preprocessing methods,
        tokenization methods, token weighting methods and distance functions.
        The space will be the cartesian product of all options in the dict.
        See ./join_function_space/options.py for defining join functions using
        a dict.
        (3) Use a list of customized JoinFunction objects.

    n_jobs : int, default=1
        Number of CPU cores used. -1 means using all processors.

    verbose: bool, default=False
        Whether to print logging

    cache_dir: string, default="autofj_temp"
        Directory for cache
    """

    def __init__(self,
                 join_function_space="autofj_sm",
                 n_jobs=-1,
                 verbose=False,
                 cache_dir="autofj_temp"):
        if type(join_function_space) == str:
            builtin_jfs = {
                "autofj_sm": autofj_sm,
                "autofj_md": autofj_md,
                "autofj_lg": autofj_lg,
            }

            if join_function_space not in builtin_jfs:
                raise Exception("{} is not a valid join function space"
                                .format(join_function_space))

            self.join_functions = self._get_autofj_join_functions(
                builtin_jfs[join_function_space])
            self.use_builtin_fj_space = True
        elif type(join_function_space) == dict:
            self.join_functions = self._get_autofj_join_functions(
                join_function_space)
            self.use_builtin_fj_space = True
        else:
            self.join_functions = join_function_space
            self.use_builtin_fj_space = False

        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.cache_dir = cache_dir

    def compute_distance(self, left, right, LL_blocked, LR_blocked):
        """Compute distance between each record pair in the blocked table

        Parameters
        ----------
        left: pd.DataFrame
            The left table. The id column is named as autofj_id.

        right: pd.DataFrame
            The right table. The id column is named as autofj_id.

        LL_blocked: pd.DataFrame
            The blocked table of left-left self-join, i.e., candidate set.
            The id columns are named as autofj_id_l and autofj_id_r,
            corresponding to the id column of left and another left table,
            respectively.

        LR_blocked: pd.DataFrame
            The blocked table of left-right join, i.e., candidate set.
            The id columns are named as autofj_id_l and autofj_id_r,
            corresponding to the id column of left and right table, respectively.

        Return:
        -------
        LR_distance: dict {string: pd.DataFrame}
            A dict of distance tables measured by different join functions. The
            key of dict is the name of the join function. The value is a
            distance table that contains distance between tuple pairs for each
            column (see the example below).

            Example of distance table:
            | autofj_id_l | autofj_id_r | column 1 | column 2 | column 3 |
            |------------------------------------------------------------|
            |      0      |      1      |    0.5   |     0.2  |     1    |
            |      0      |      2      |    0.3   |     0.4  |    0.7   |
        """
        # sort by ids
        LL_blocked = LL_blocked[["autofj_id_l", "autofj_id_r"]] \
            .sort_values(by=["autofj_id_r", "autofj_id_l"]) \
            .reset_index(drop=True)
        LR_blocked = LR_blocked[["autofj_id_l", "autofj_id_r"]] \
            .sort_values(by=["autofj_id_r", "autofj_id_l"]) \
            .reset_index(drop=True)

        # build cache for computing distance
        if self.use_builtin_fj_space:
            self.build_cache(left, right, self.cache_dir)

        # compute distance for each column
        column_names = [c for c in left.columns if c != "autofj_id"]
        args = []
        for c in column_names:
            for jf in self.join_functions:
                args.append((c, jf))

        func = partial(self._compute_column_distance, left=left, right=right,
                       LL_blocked=LL_blocked, LR_blocked=LR_blocked,
                       cache_dir=self.cache_dir)

        if self.n_jobs == 1:
            column_distances = [func(arg) for arg in args]
        else:
            with Pool(self.n_jobs) as pool:
                column_distances = pool.map(func, args)

        # remove cache
        if self.use_builtin_fj_space:
            self.remove_cache(self.cache_dir)

        # get distance table for each join function
        LL_distance = {}
        LR_distance = {}

        for i, (c, jf) in enumerate(args):
            jf_name = jf.name
            if jf_name not in LL_distance:
                LL_distance[jf_name] = {
                    "autofj_id_l": LL_blocked["autofj_id_l"].values,
                    "autofj_id_r": LL_blocked["autofj_id_r"].values}
                LR_distance[jf_name] = {
                    "autofj_id_l": LR_blocked["autofj_id_l"].values,
                    "autofj_id_r": LR_blocked["autofj_id_r"].values}

            LL_d, LR_d = column_distances[i]
            LL_distance[jf_name][c] = LL_d
            LR_distance[jf_name][c] = LR_d

        for jf_name in LL_distance.keys():
            LL_distance[jf_name] = pd.DataFrame(LL_distance[jf_name])
            LR_distance[jf_name] = pd.DataFrame(LR_distance[jf_name])

        return LL_distance, LR_distance

    def _compute_column_distance(self, arg, left, right, LL_blocked, LR_blocked,
                                cache_dir):
        """Compute distance for a column using a join function


        Parameters
        ----------
        arg: tuple
            A tuple that consists of column_name and join function object

        Returns
        -------
        LL_d: pd.Series
            Distance of tuple pairs in LL_blocked

        LR_d: pd.Series
            Distance of tuple pairs in LR_blocked
        """
        c, jf = arg
        if self.verbose:
            print_log("{}: Compute distances for column {}.".format(jf.name, c))

        # get subsets of column c from tables
        l = left[["autofj_id", c]].rename(columns={c: "value"})
        r = right[["autofj_id", c]].rename(columns={c: "value"})

        cache_dir = os.path.join(cache_dir, c)

        # compute distances for column c
        LL_d, LR_d = jf.compute_distance(l, r, LL_blocked, LR_blocked, cache_dir)
        return LL_d, LR_d

    def build_cache(self, left, right, cache_dir="autofj_temp"):
        """ Build cache for computing distance

        Parameters
        ----------
        left: pd.DataFrame
            The left table. The id column is named as autofj_id.

        right: pd.DataFrame
            The right table. The id column is named as autofj_id.

        cache_dir: string
            Directory to save cache

        """
        # indexing
        left = left.set_index("autofj_id")
        right = right.set_index("autofj_id")

        # build cache for each column of left and right table
        column_names = [c for c in left.columns if c != "autofj_id"]
        func = partial(self._build_cache_column, left=left, right=right,
                       cache_dir=cache_dir)
        args = []
        for c in column_names:
            args.append((c, "left"))
            args.append((c, "right"))

        with Pool(self.n_jobs) as pool:
            pool.map(func, args)

    def _build_cache_column(self, column_table, left, right, cache_dir):
        """Built cache on one column

        Parameters
        ----------
        column: string
            The column to build cache
        """
        # get data under processing
        column, table = column_table
        if table == "left":
            X = left[column]
        else:
            X = right[column]

        preprocess_results = {}
        tokenize_results = {}
        token_weights = {}
        column_cache_dir = os.path.join(cache_dir, column)

        # cache results
        for jf in self.join_functions:
            # get cache paths
            pre_path, pre_token_path, weight_path \
                = jf.get_cache_paths(column_cache_dir, table)
            p = jf.preprocess_method
            t = jf.tokenize_method
            w = jf.token_weight_method

            if p not in preprocess_results:
                X_pre = jf.preprocess(X, cache_path=pre_path)
                preprocess_results[p] = X_pre
            else:
                X_pre = preprocess_results[p]

            if (p, t) not in tokenize_results:
                X_pre_token = jf.tokenize(X_pre, cache_path=pre_token_path)
                tokenize_results[(p, t)] = X_pre_token
            else:
                X_pre_token = tokenize_results[(p, t)]

            if table == "left" and (p, t, w) not in token_weights:
                weights = jf.compute_token_weights(X_pre_token,
                                                   cache_path=weight_path)
                token_weights[(p, t, w)] = weights

    def remove_cache(self, cache_dir="autofj_temp"):
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    def _get_autofj_join_functions(self, options):
        """ Get a space of built-in join functions with different options. The
        space will be formed by making cartesian product on all different
        options.

        Parameters
        ----------
        options: dict
            The options of preprocess methods, tokenization methods, token weights
            and distance functions. See option.py for examples. The dict contains
            the following key-values:
            - preprocess_methods: list of string
                A list of preprocess methods defined in preprocessor.py
            - tokenize_methods: list of string
                A list of tokenization methods defined in tokenizer.py
            - token_weights: list of string
                A list of tokenization methods defined in token_weights.py
            - char_distance_functions: list of string
                A list of character-based distance functions defined in
                distance_function.py
            - set_distance_functions: list of string
                A list of set-based distance functions defined in
                distance_function.py

        Return
        ------
        join_functions: list of AutoFJJoinFunction object
            A list of join functions.
        """
        join_functions = []
        # join functions using set-based distance functions
        for p in options["preprocess_methods"]:
            for t in options["tokenize_methods"]:
                for w in options["token_weights"]:
                    for d in options["set_distance_functions"]:
                        jf = AutoFJJoinFunction(p, t, w, d)
                        join_functions.append(jf)

        # join functions using char-based distance functions
        for p in options["preprocess_methods"]:
            for d in options["char_distance_functions"]:
                jf = AutoFJJoinFunction(p, None, None, d)
                join_functions.append(jf)
        return join_functions