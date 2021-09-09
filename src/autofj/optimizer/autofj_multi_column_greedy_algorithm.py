import pandas as pd
import numpy as np
from multiprocessing import Pool
from .autofj_single_column_greedy_algorithm import AutoFJGreedyAlgorithm
from ..utils import print_log
import os

class AutoFJMulticolGreedyAlgorithm(object):
    """Greedy algorithm for multi-column datasets. Select optimal columns and
    column weights using column forward selection.

    Parameters
    ----------
    LL_distance: dict
        Distance of tuple pairs in LL tables measured by different join
        functions. The distance is saved in a dict. The key is the name
        of join functions. The value is a table of distance between
        tuple pairs on different columns. The first two columns in the
        table are "autofj_id_l", "autofj_id_r". The remaining columns
        are distance on different columns.

    LR_distance: dict
        Distance of tuple pairs in LR tables measured by different join
        functions. The distance is saved in a dict. The key is the name
        of join functions. The value is a pd.DataFrame of distance between
        tuple pairs on different columns. The first two columns in the
        table are "autofj_id_l", "autofj_id_r". The remaining columns
        are distance on different columns.

    precision_target: float
        Precision target. This should be a float number between 0-1.

    candidate_thresholds: list
        The search space of distance threshold.

    candidate_column_weights: list
        The search space of column weights.

    n_jobs : int, default=1
        Number of CPU cores used. -1 means using all processors.

    verbose: bool, default=False
        Whether to print logging

    """
    def __init__(self,
                 LL_distance,
                 LR_distance,
                 precision_target,
                 candidate_thresholds,
                 candidate_column_weights,
                 n_jobs=-1,
                 verbose=False):
        self.precision_target = precision_target
        self.LL_distance = LL_distance
        self.LR_distance = LR_distance
        self.join_functions = sorted(list(LL_distance.keys()))
        self.candidate_thresholds = candidate_thresholds
        self.candidate_column_weights = candidate_column_weights
        self.verbose = verbose
        self.column_names = []
        for c in LL_distance[self.join_functions[0]].columns:
            if c != "autofj_id_l" and c != "autofj_id_r":
                self.column_names.append(c)
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()

    def run(self):
        """Running forward selection algorithm"""
        best_reward = float("-inf")
        best_LR_joins = None
        best_columns = []
        best_weights = None
        best_join_config = None
        best_column_weights = None

        for i in range(len(self.column_names)):
            # get the best result after adding one column
            columns, weights, join_config, LR_joins, reward \
                = self.forward_selection(best_columns, best_weights)

            # if the reward stops increasing by adding columns, terminates
            if reward <= best_reward:
                break

            # save best result
            best_columns, best_weights, best_join_config, best_LR_joins, best_reward \
                = columns, weights, join_config, LR_joins, reward

            if best_join_config is not None:
                best_column_weights = self.get_column_weights(best_columns,
                                                              best_weights)
            if self.verbose:
                print_log("Best column_weight: {}:{}, Best reward: {}"
                      .format(",".join(best_columns),
                              ",".join([str(w) for w in best_weights]),
                              best_reward))

        return best_column_weights, best_join_config, best_LR_joins

    def forward_selection(self, base_columns, base_weights):
        """Do one step forward selection. Adding one column from the remaining
        columns, get the best column and weights.

        Parameters
        ----------
        base_columns: list
            Old columns (best column from the last iteration)

        base_weights: list
            Old weights (best weight from the last iteration)

        Returns
        -------
        best_columns: list
            Best columns after adding one column

        best_weights: list
            Best weights after adding one column

        best_pred: list
            LR prediction

        best_reward: float
            resulting reward (profit or estimated recall) given the best
            columns and weights
        """
        # get all candidate column weights
        column_weights_cands = self.get_column_weights_cands(base_columns,
                                                             base_weights)

        if self.n_jobs == 1:
            results = []
            for column_weights in column_weights_cands:
                res = self.run_one_weight(column_weights)
                results.append(res)
        else:
            with Pool(self.n_jobs) as pool:
                results = pool.map(self.run_one_weight, column_weights_cands)

        best_reward = float("-inf")
        best_weights = None
        best_LR_joins = None
        best_join_config = None
        best_columns = None

        for i, (LR_joins, reward, config_selected) in enumerate(results):
            if reward > best_reward:
                best_reward = reward
                best_columns = list(column_weights_cands[i].keys())
                best_weights = list(column_weights_cands[i].values())
                best_LR_joins = LR_joins
                best_join_config = config_selected
        return best_columns, best_weights, best_join_config, best_LR_joins, \
               best_reward

    def get_column_weights_cands(self, base_columns, base_weights):
        """Get candidate column weights by adding one column into old columns

        Parameters
        ----------
        base_columns: list
            Old columns (best column from the last iteration)

        base_weights: list
            Old weights (best weight from the last iteration)

        Returns
        -------
        column_weights_cands: list
            A list of candidate column weights (dict)

        """
        """ get column-weight candidates"""
        column_weights_cands = []
        for c in self.column_names:
            if c in base_columns:
                continue
            columns = base_columns + [c]

            # get search space (weights candidates) for new weights
            new_weights = self.get_new_weights(base_weights)

            for weights in new_weights:
                column_weights = self.get_column_weights(columns, weights)
                column_weights_cands.append(column_weights)
        return column_weights_cands

    def run_one_weight(self, column_weights):
        # if self.verbose:
        #     print_log("Run greedy algorithm with column weights"
        #           .format(column_weights)

        LL_w = self.get_weighted_distance(self.LL_distance, column_weights)
        LR_w = self.get_weighted_distance(self.LR_distance, column_weights)
        optimizer = AutoFJGreedyAlgorithm(LL_w,
                                          LR_w,
                                          self.precision_target,
                                          self.candidate_thresholds,
                                          n_jobs=self.n_jobs)
        LR_joins, config_selected = optimizer.run()
        reward = optimizer.get_reward()
        return LR_joins, reward, config_selected

    def get_new_weights(self, old_weights):
        """Get column weight search space. Keeping ratios of old weights fixed,
           append weight for the new column.

        Parameters
        ----------
        old_weights: list
             Weights of old columns

        Return
        ------
        column_weights: list
            A list of new weights. In each new weight, the last one is the new
            weight, the others are old weights. The ratio between old weights
            is fixed.
            Example: old_weights: [0.5, 0.5]
                column_weights = [[0.45, 0.45, 0.1],
                                  [0.40, 0.40, 0.2],
                                  [0.35, 0.35, 0.3],
                                  ...]
        """
        # if old weight is empty, new weight is [1].
        if old_weights is None:
            return np.array([[1]])

        # add weight for new column
        column_weights = []
        for nw in self.candidate_column_weights[1:-1]:
            new_w = np.array([w * nw for w in old_weights] + [1 - nw])
            column_weights.append(new_w)

        return column_weights

    def get_weighted_distance(self, LL_distance, column_weights):
        """LL_w: {config: [lid, rid, distance]}"""
        LL_w = {}
        for config, distance in LL_distance.items():
            columns = [c for c in distance.columns if
                       c not in ["autofj_id_l", "autofj_id_r"]]
            weights = []
            for c in columns:
                if c in column_weights:
                    weights.append(column_weights[c])
                else:
                    weights.append(0)
            weights = np.array(weights).reshape(-1, 1)
            weighted_dist = distance[columns].values.dot(weights).ravel()
            LL_w[config] = pd.DataFrame(
                {"autofj_id_l": distance["autofj_id_l"],
                 "autofj_id_r": distance["autofj_id_r"],
                 "distance": weighted_dist})
        return LL_w

    def get_column_weights(self, columns, weights):
        column_weights = {}
        for c, w in zip(columns, weights):
            column_weights[c] = w
        return column_weights