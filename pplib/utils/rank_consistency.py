import copy
import math

import numpy as np
import scipy.stats


def pearson(true_vector, pred_vector):
    n = len(true_vector)
    # simple sums
    sum1 = sum(float(true_vector[i]) for i in range(n))
    sum2 = sum(float(pred_vector[i]) for i in range(n))
    # sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in true_vector])
    sum2_pow = sum([pow(v, 2.0) for v in pred_vector])
    # sum up the products
    p_sum = sum([true_vector[i] * pred_vector[i] for i in range(n)])
    # 分子num，分母den
    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt(
        (sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
    if den == 0:
        return 0.0
    return num / den


def kendalltau(true_vector, pred_vector):
    tau, p_value = scipy.stats.kendalltau(true_vector, pred_vector)
    return tau


def spearman(true_vector, pred_vector):
    coef, p_value = scipy.stats.spearmanr(true_vector, pred_vector)
    return coef


def rank_difference(true_vector, pred_vector):
    # assert true_vector
    assert len(true_vector) == len(pred_vector)

    def get_rank(vector):
        v = np.array(vector)
        v_ = copy.deepcopy(v)
        v_.sort()
        rank = []

        for i in v:
            rank.append(list(v_).index(i))
        return rank

    rank1 = get_rank(true_vector)
    rank2 = get_rank(pred_vector)

    length = len(true_vector)

    sum_rd = 0.
    for i in range(length):
        sum_rd += rank1[i] - rank2[i]

    return sum_rd / length
