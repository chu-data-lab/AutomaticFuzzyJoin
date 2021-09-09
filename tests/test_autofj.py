import pandas as pd
import time
import os
from autofj import AutoFJ
from autofj.datasets import load_data

def evaluate(pred_joins, gt_joins):
    """ Evaluate the performance of fuzzy joins

    Parameters
    ----------
    pred_joins: list
        A list of tuple pairs (id_l, id_r) that are predicted to be matches

    gt_joins:
        The ground truth matches

    Returns
    -------
    precision: float
        Precision score

    recall: float
        Recall score

    f1: float
        F1 score
    """
    pred = {(l, r) for l, r in pred_joins}
    gt = {(l, r) for l, r in gt_joins}
    tp = pred.intersection(gt)

    precision = len(tp) / len(pred)
    recall = len(tp) / len(gt)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def test_autofj(dataset):
    left, right, gt = load_data(dataset)
    autofj = AutoFJ(verbose=True)
    LR_joins = autofj.join(left, right, id_column="id")
    
    print(LR_joins)
    gt_joins = gt[["id_l", "id_r"]].values
    LR_joins = LR_joins[["id_l", "id_r"]].values
    p, r, f1 = evaluate(LR_joins, gt_joins)
    print("Precision:", p, "Recall:", r, "F1:", f1)

if __name__ == '__main__':
    test_autofj("TennisTournament")

