import pandas as pd
from nltk.stem.porter import PorterStemmer
import re


class NegativeRule(object):
    """Negative rules"""

    def __init__(self, left, right, id_column):
        self.left = self._preprocess(left, id_column)
        self.right = self._preprocess(right, id_column)
        self.id_column = id_column
        self.negative_rules = set()

    def get_tokens_diff(self, l_tokens, r_tokens):
        # get difference of l_token set and r_token set
        l_diff = [l - r for l, r in zip(l_tokens, r_tokens)]
        r_diff = [r - l for l, r in zip(l_tokens, r_tokens)]
        return l_diff, r_diff

    def _preprocess(self, df, id_column):
        """ Preprocess the records: (1) concatenate all columns. (2) lowercase,
        remove punctuation and do stemming (3) split by space

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
            new_x = set(stem_x)
            new_value.append(new_x)

        id_df = df[id_column].values
        result = pd.DataFrame({id_column: id_df, "value": new_value})
        return result

    def learn(self, LL_blocked):
        """Learn opposite rules from LL"""
        # merge LL with left
        LL = self._merge(self.left, self.left, LL_blocked)

        # get token difference
        l_diff, r_diff = self.get_tokens_diff(LL["value_l"].values,
                                              LL["value_r"].values)

        # get rules: (l_token, r_token) that have one different token from each other
        for l, r, l_set in zip(l_diff, r_diff, LL["value_l"]):
            if len(l) == 1 and len(r) == 1 and len(l_set) != 1:
                self.negative_rules.add((list(l)[0], list(r)[0]))
                self.negative_rules.add((list(r)[0], list(l)[0]))

        # print(self.negative_rules)
        # raise

    def _merge(self, left, right, LR_blocked):
        id_column = self.id_column
        LR = LR_blocked[[id_column + "_l", id_column + "_r"]]
        LR = LR.merge(left, left_on=id_column + "_l", right_on=id_column)\
               .drop(columns=id_column) \
               .merge(right, left_on=id_column + "_r", right_on=id_column,
                   suffixes=("_l", "_r"))\
               .drop(columns=id_column)
        return LR

    def apply(self, LR_blocked):
        """Apply opposite rule on LR blocked"""
        # merge LR with left, right
        LR = self._merge(self.left, self.right, LR_blocked)

        # get token difference
        l_diff, r_diff = self.get_tokens_diff(LR["value_l"].values,
                                              LR["value_r"].values)

        # apply rule
        mask = []
        for lid, rid, l_d, r_d in zip(LR["autofj_id_l"].values,
                                      LR["autofj_id_r"].values,
                                      l_diff,
                                      r_diff):
            pairs = [(l, r) for l in l_d for r in r_d]
            meet_rule = any([p in self.negative_rules for p in pairs])
            mask.append(not meet_rule)

        LR_blocked = LR[mask][["autofj_id_l", "autofj_id_r"]]
        return LR_blocked
