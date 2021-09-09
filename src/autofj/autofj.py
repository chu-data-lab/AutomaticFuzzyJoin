from .join_function_space.autofj_join_function_space import AutoFJJoinFunctionSpace
from .blocker.autofj_blocker import AutoFJBlocker
from .optimizer.autofj_multi_column_greedy_algorithm import \
    AutoFJMulticolGreedyAlgorithm
import pandas as pd
from .utils import print_log
import os
from .negative_rule import NegativeRule
import numpy as np


class AutoFJ(object):
    """
    AutoFJ automatically produces record pairs that approximately match in 
    two tables L and R. It proceeds to configure suitable parameters 
    automatically, which when used to fuzzy-join L and R, meets the 
    user-specified precision target, while maximizing recall.
    
    AutoFJ attempts to solve many-to-one join problems, where each record in R
    will be joined with at most one record in L, but each record in L can be 
    joined with multiple records in R. In AutoFJ, L refers to a reference 
    table, which is assumed to be almost "duplicate-free".

    Parameters
    ----------
    precision_target: float, default=0.9
        Precision target.

    join_function_space: string or dict or list of objects, default="autofj_sm"
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

    distance_threshold_space: int or list, default=50
        The number of candidate distance thresholds or a list of candidate
        distance thresholds in the space.  If the number of distance thresholds
        (integer) is given, distance thresholds are spaced evenly from 0 to 1.
        Otherwise, it should be a list of floats from 0 to 1.

    column_weight_space: int or list, default=10
        The number of candidate column weights or a list of candidate
        column weights in the space. If the number of column weights
        (integer) is given, column weights are spaced evenly from 0 to 1.
        Otherwise, it should be a list of floats from 0 to 1.

    blocker: a Blocker object or None, default None
        A Blocker object that performs blocking on two tables. If None, use 
        the built-in blocker. For customized blocker, see Blocker class.

    n_jobs : int, default=-1
        Number of CPU cores used. -1 means using all processors.

    verbose: bool, default=False
        Whether to print logging
    """

    def __init__(self,
                 precision_target=0.9,
                 join_function_space="autofj_sm",
                 distance_threshold_space=50,
                 column_weight_space=10,
                 blocker=None,
                 n_jobs=-1,
                 verbose=False):
        self.precision_target = precision_target
        self.join_function_space = join_function_space

        if type(distance_threshold_space) == int:
            self.distance_threshold_space = list(
                np.linspace(0, 1, distance_threshold_space))
        else:
            self.distance_threshold_space = distance_threshold_space

        if type(column_weight_space) == int:
            self.column_weight_space = list(
                np.linspace(0, 1, column_weight_space))
        else:
            self.column_weight_space = column_weight_space

        if blocker is None:
            self.blocker = AutoFJBlocker(n_jobs=n_jobs)
        else:
            self.blocker = blocker

        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.verbose = verbose

    def join(self, left_table, right_table, id_column, on=None):
        """Join left table and right table.

        Parameters
        ----------
        left_table: pd.DataFrame
            Reference table. The left table is assumed to be almost
            duplicate-free, which means it has no or only few duplicates.

        right_table: pd.DataFrame
            Another input table.

        id_column: string
            The name of id column in the two tables. This column will not be
            used to join two tables.

        on: list or None
            A list of column names (multi-column fuzzy join) that the two tables
            will be joined on. If None, two tables will be joined on all columns
            that exist in both tables, excluding the id column.

        Returns:
        --------
            result: pd.DataFrame
                A table of joining pairs. The columns of left table are
                suffixed with "_l" and the columns of right table are suffixed
                with "_r"
        """
        left = left_table.copy(deep=True)
        right = right_table.copy(deep=True)

        # create internal id columns (use internal ids)
        left["autofj_id"] = range(len(left))
        right["autofj_id"] = range(len(right))

        # remove original ids
        left.drop(columns=id_column, inplace=True)
        right.drop(columns=id_column, inplace=True)

        # get names of columns to be joined
        if on is None:
            on = sorted(list(set(left.columns).intersection(right.columns)))
        left = left[on]
        right = right[on]

        # do blocking
        if self.verbose:
            print_log("Start blocking")
        LL_blocked = self.blocker.block(left, left, "autofj_id")
        LR_blocked = self.blocker.block(left, right, "autofj_id")

        # remove equi-joins on LL
        LL_blocked = LL_blocked[
            LL_blocked["autofj_id_l"] != LL_blocked["autofj_id_r"]]

        # learn and apply negative rules
        nr = NegativeRule(left, right, "autofj_id")
        nr.learn(LL_blocked)
        LR_blocked = nr.apply(LR_blocked)

        # create join function space
        jf_space = AutoFJJoinFunctionSpace(self.join_function_space,
                                           n_jobs=self.n_jobs)

        # compute distance
        if self.verbose:
            print_log("Start computing distances. Size of join function space: {}"
                      .format(len(jf_space.join_functions)))

        LL_distance, LR_distance = jf_space.compute_distance(left,
                                                             right,
                                                             LL_blocked,
                                                             LR_blocked)

        # run greedy algorithm
        if self.verbose:
            print_log("Start running greedy algorithm.")

        optimizer = AutoFJMulticolGreedyAlgorithm(
            LL_distance,
            LR_distance,
            precision_target=self.precision_target,
            candidate_thresholds=self.distance_threshold_space,
            candidate_column_weights=self.column_weight_space,
            n_jobs=self.n_jobs
        )

        self.selected_column_weights, self.selected_join_configs, LR_joins = \
            optimizer.run()

        if LR_joins is None:
            print("Warning: The precision target cannot be achieved.",
                  "Try a lower precision target or a larger space of join functions,",
                  "distance thresholds and column weights.")
            LR_joins = pd.DataFrame(columns=[c+"_l" for c in left_table.columns]+
                                            [c+"_r" for c in right_table.columns])
            return LR_joins

        # merge with original left and right tables
        left_idx = [l for l, r in LR_joins]
        right_idx = [r for l, r in LR_joins]
        L = left_table.iloc[left_idx].add_suffix("_l").reset_index(drop=True)
        R = right_table.iloc[right_idx].add_suffix("_r").reset_index(drop=True)
        result = pd.concat([L, R], axis=1).sort_values(by=id_column + "_r")
        return result
