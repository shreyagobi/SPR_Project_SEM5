# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist
from utils import mfcc, get_mfcc_sequences, get_test_files, save_models


@njit
def _dtw_dp(cost):
    """Numba-accelerated dynamic programming pass over pairwise frame cost matrix."""
    m1, m2 = cost.shape
    dp = np.empty((m1 + 1, m2 + 1), dtype=np.float64)
    dp[0, :] = np.inf
    dp[:, 0] = np.inf
    dp[0, 0] = 0.0
    for i in range(m1):
        for j in range(m2):
            v0 = dp[i, j]
            v1 = dp[i + 1, j]
            v2 = dp[i, j + 1]
            best = v0 if v0 <= v1 else v1
            if v2 < best:
                best = v2
            dp[i + 1, j + 1] = cost[i, j] + best
    return dp


def dtw(M1, M2):
    """
    Compute Dynamic Time Warping (DTW) of two MFCC sequences.
    :param M1, M2: two MFCC sequences (T1, D) and (T2, D)
    :return: (normalized_min_distance, (path_1, path_2))
    """
    M1_len = len(M1)
    M2_len = len(M2)
    cost = cdist(M1, M2, metric='euclidean')
    dp = _dtw_dp(cost)

    if M1_len == 1:
        path = np.zeros(M2_len, dtype=int), np.arange(M2_len)
    elif M2_len == 1:
        path = np.arange(M1_len), np.zeros(M1_len, dtype=int)
    else:
        i, j = M1_len - 1, M2_len - 1
        path_1, path_2 = [i], [j]
        while (i > 0) or (j > 0):
            v0 = dp[i, j]
            v1 = dp[i, j + 1]
            v2 = dp[i + 1, j]
            if v0 <= v1 and v0 <= v2:
                i -= 1
                j -= 1
            elif v1 <= v2:
                i -= 1
            else:
                j -= 1
            path_1.append(i)
            path_2.append(j)
        path_1.reverse()
        path_2.reverse()
        path = np.array(path_1), np.array(path_2)

    return float(dp[M1_len, M2_len] / (M1_len + M2_len)), path


def train_model_dtw(train_dir, max_templates_per_digit=8, cmvn=False):
    """
    Store up to `max_templates_per_digit` evenly-spaced templates per digit 0..9.
    Multi-template DTW avoids blurring multiple speakers into a single averaged template.
    """
    sequences = get_mfcc_sequences(train_dir, cmvn=cmvn)
    model = []
    for digit in range(10):
        seqs = sequences[digit]
        if len(seqs) > max_templates_per_digit:
            indices = np.linspace(0, len(seqs) - 1, max_templates_per_digit, dtype=int)
            seqs = [seqs[idx] for idx in indices]
        model.append(seqs)
    return model


def predict_from_features(model, mfcc_feat):
    """
    Predict digit using nearest-template DTW across stored templates per digit.
    :return: (predicted_digit_int, margin_over_second_best, negative_distances_array)
    """
    min_dists = np.zeros(len(model), dtype=float)
    for digit, templates in enumerate(model):
        if isinstance(templates, list):
            min_dists[digit] = min(dtw(t, mfcc_feat)[0] for t in templates)
        else:
            min_dists[digit] = dtw(templates, mfcc_feat)[0]

    sorted_idx = np.argsort(min_dists)
    best = int(sorted_idx[0])
    margin = float(min_dists[sorted_idx[1]] - min_dists[best])
    return best, margin, -min_dists


def predict_dtw(model, file_path, cmvn=False):
    """Predict digit string for a .wav file."""
    mfcc_feat = mfcc(file_path, cmvn=cmvn)
    digit, _, _ = predict_from_features(model, mfcc_feat)
    return str(digit)


if __name__ == '__main__':
    train_dir = "./processed_train_records"
    test_dir = "./processed_test_records"
    model = train_model_dtw(train_dir)
    save_models(model, "./models/dtw_templates.joblib")
    count = 0
    pred_true = 0
    for file_path, ground_truth in get_test_files(test_dir):
        count += 1
        pred = predict_dtw(model, file_path)
        if int(pred) == ground_truth:
            pred_true += 1
    print("----------Dynamic Time Warping (DTW)----------")
    print(f"Train num: 160, Test num: {count}, Predict true num: {pred_true}")
    print(f"Accuracy: {pred_true / count:.2f}")
