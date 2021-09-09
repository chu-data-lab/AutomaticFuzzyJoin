import numpy as np
import time
import os
from copy import deepcopy
from ..utils import print_log


class AutoFJGreedyAlgorithm(object):
    """Greedy Algorithm for single column datasets. Select a union of join
    configurations that maximize the recall while keeping the precision above
    the precision target.

    Parameters
    ----------
    LL_distance: dict
        Distance of tuple pairs in LL tables on one column measured by different
        join functions. The distance is saved in a dict. The key is the name
        of join functions. The value is a table of distance between tuple pairs
        on one column. The first two columns in the table are "autofj_id_l",
        "autofj_id_r". The third column is the "distance".

    LR_distance: dict
        Distance of tuple pairs in LR tables on one column measured by different
        join functions. The distance is saved in a dict. The key is the name
        of join functions. The value is a table of distance between tuple pairs
        on one column. The first two columns in the table are "autofj_id_l",
        "autofj_id_r". The third column is the "distance"

    precision_target: float
        Precision target.

    unscaled_candidate_thresholds: list
        Candidates of thresholds

    n_jobs : int, default=1
        Number of CPU cores used. -1 means using all processors.
    """

    def __init__(self,
                 LL_distance,
                 LR_distance,
                 precision_target,
                 unscaled_candidate_thresholds,
                 patience=0.05,
                 verbose=False,
                 n_jobs=-1):
        self.target = precision_target
        self.full_LR_distance = LR_distance
        self.LL_distance = self.groupby_rid(LL_distance)
        self.LR_distance = self.keep_min_distance(LR_distance)
        self.join_functions = sorted(list(LL_distance.keys()))
        self.candidate_thresholds = self.scale_candidate_thresholds(
            unscaled_candidate_thresholds)
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.precision_cache = self.precompute_precision()
        self.patience = patience
        self.verbose=verbose

    def keep_min_distance(self, LR_distance):
        # for each rid only keep lid with the smallest distance
        LR_small = {}
        for config, LR in LR_distance.items():
            LR_s = LR.sort_values(by=["distance", "autofj_id_l"],
                                  ascending=True).drop_duplicates(
                subset="autofj_id_r", keep="first")
            LR_s = LR_s.sort_values(by=["autofj_id_r", "autofj_id_l"])
            LR_small[config] = LR_s
        return LR_small

    def groupby_rid(self, LL_distance):
        """Group the distance table by rid. lids and distance for one rid will
         be put in one table.
         """
        LL_group = {}
        for config, LL in LL_distance.items():
            rids = LL["autofj_id_r"].values
            break_indices = np.argwhere(np.diff(rids) > 0).ravel()
            values = np.split(LL, break_indices + 1)
            keys = [rids[0]] + rids[break_indices + 1].tolist()
            LL_dict = {keys[i]: values[i] for i in range(len(values))}
            LL_group[config] = LL_dict
        return LL_group

    def scale_candidate_thresholds(self, unscaled_thresholds):
        """Scale candidate thresholds for each join function such that the
           threshold is in the range of [min, max] of LR distance.
        """
        candidate_thresholds = {}
        unscaled_thresholds = np.array(unscaled_thresholds)

        for jf in self.join_functions:
            max_d = self.LR_distance[jf]["distance"].values.max()
            min_d = self.LR_distance[jf]["distance"].values.min()
            cand_thresh = unscaled_thresholds * (max_d - min_d) + min_d
            candidate_thresholds[jf] = set(cand_thresh.tolist())
        return candidate_thresholds

    def precompute_precision(self):
        """ Precompute precision of each LR join for every join configuration
        The precision is computed as 1/ #L-L joins
        Returns
        -------
        precision: dict
            The precision of each LR join for every join configuration. The key
            is the join configuration (join function, threshold). The value is
            a dict, where the key is rid and value is a tuple (lid, precision).
        """
        precision = {}
        for jf in self.join_functions:
            LL = self.LL_distance[jf]
            LR = self.LR_distance[jf]
            for thresh in self.candidate_thresholds[jf]:
                prec = {}

                for lid, rid, d in LR.values:
                    if d > thresh:
                        continue
                    # compute precision as 1/ #L-L joins
                    num_LL_joins = self.get_num_LL_joins(LL, lid, thresh)
                    prec[rid] = (lid, 1 / num_LL_joins)

                precision[(jf, thresh)] = prec

        return precision

    def get_num_LL_joins(self, LL, proxy, thresh):
        """ Number of L-L joins of proxy (the l record closest to each R),
        which is the number of L records that have distance smaller than
        2 * threshold to the proxy"""
        if proxy not in LL:
            return 1
        else:
            lid_df = LL[proxy]
            mask = lid_df["distance"] <= 2 * thresh
            num_proxy_joins = mask.sum() + 1
            return num_proxy_joins

    def compute_profit(self, added_config):
        """ Compute profit of adding one join configuration to the union. The
        profit is defined as true positive (TP) / false positive (FP).

        Parameters
        ----------
        added_config: tuple
            Join configuration to be added (join function, threshold)

        Returns
        -------
        profit: float
            Estimated TP / Estimated FP
        """
        # extract precision from cache
        precision = self.precision_cache[added_config]

        # compute the change of TP and number of joins by adding the new config
        TP = self.running_TP
        n_joins = self.running_n_joins

        for r, (l, p) in precision.items():
            if r in self.running_l_cands:
                # If r has been joined before, conflict may occur.
                # To resolve conflicts, use the join with the highest precision.
                # Only this join is counted toward TP. But other joins are
                # preserved in FP to penalize on conflicts.
                l_cands = self.running_l_cands[r]
                old_p = self.running_local_prec[r]
                delta_TP = max(old_p, p) - old_p
                delta_n_joins = 0 if l in l_cands else 1
            else:
                delta_TP = p
                delta_n_joins = 1

            TP += delta_TP
            n_joins += delta_n_joins

        FP = n_joins - TP
        avg_prec = TP / (n_joins + 1e-9)

        if n_joins == self.running_n_joins:
            # ensure some new records are joined
            profit = float("-inf")
        else:
            profit = TP / (FP + 1e-9)

        return profit, avg_prec

    def update_selection(self, config, avg_prec):
        """Updated selected join configuration"""
        self.running_configs.append(config)

        for r, (l, p) in self.precision_cache[config].items():
            if r in self.running_l_cands:
                l_cands = self.running_l_cands[r]
                old_p = self.running_local_prec[r]

                if p > old_p:
                    self.running_local_prec[r] = p
                    self.running_LR_joins[r] = l
                    delta_TP = p - old_p
                else:
                    delta_TP = 0

                if l in l_cands:
                    delta_n_joins = 0
                else:
                    self.running_l_cands[r].add(l)
                    delta_n_joins = 1
            else:
                self.running_l_cands[r] = {l}
                self.running_local_prec[r] = p
                self.running_LR_joins[r] = l
                delta_TP = p
                delta_n_joins = 1

            self.running_TP += delta_TP
            self.running_n_joins += delta_n_joins

        if avg_prec > self.target:
            self.global_prec = avg_prec
            self.config_selected = deepcopy(self.running_configs)
            self.LR_joins = [(l, r) for r, l in self.running_LR_joins.items()]

    def run(self):
        """Run greedy algorithm"""
        self.running_l_cands = {}
        self.running_local_prec = {}
        self.running_configs = []
        self.running_LR_joins = {}
        self.running_n_joins = 0
        self.running_TP = 0

        self.config_selected = None
        self.LR_joins = None
        self.global_prec = None

        candidate_configs = []
        for jf in self.join_functions:
            for thresh in self.candidate_thresholds[jf]:
                candidate_configs.append((jf, thresh))

        n_iter = 1
        while len(candidate_configs) > 0:
            best_profit = float("-inf")
            best_config = None
            best_avg_prec = None

            for i, config in enumerate(candidate_configs):
                profit, avg_prec = self.compute_profit(config)

                if profit > best_profit:
                    best_profit = profit
                    best_config = config
                    best_avg_prec = avg_prec

            if best_config is None:
                break

            self.update_selection(best_config, best_avg_prec)

            if self.verbose:
                print_log("Iteration {}, precision {}, # LR joins {}".\
                                format(n_iter,
                                       best_avg_prec,
                                       len(self.running_LR_joins)))

            if best_avg_prec < self.target - self.patience:
                if self.verbose:
                    print_log("Exit because precision below target")
                break

            n_iter += 1

        return self.LR_joins, self.config_selected

    def get_full_LR_sel(self):
        """Get full LR matches instead of selecting one l for each r
           Note: Only work for processing one config at each time.
        """
        if self.config_selected is None:
            return None

        config, p = self.config_selected[0]
        full_LR = self.full_LR_distance[config]
        mask = full_LR["distance"].values <= p
        lr_sel = [(l, r) for l, r in
                  full_LR[mask][["autofj_id_l", "autofj_id_r"]].values]
        return lr_sel

    def get_reward(self):
        if self.config_selected is None:
            return float("-inf")

        return self.global_prec * len(self.LR_joins)

    def get_config_selected(self):
        return self.config_selected