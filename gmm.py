# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.special import logsumexp
from utils import mfcc, get_mfcc_sequences, get_test_files, save_models


MIN_VAR = 1e-6


def log_gaussian_prob(x, means, var):
    """
    Estimate the diagonal-covariance log Gaussian probability for a single frame x.
    :param x: input vector (D,)
    :param means: component mean vector (D,)
    :param var: component variance vector (D,)
    :return: scalar log probability
    """
    var_safe = np.maximum(var, MIN_VAR)
    return (-0.5 * np.log(var_safe) - np.divide(np.square(x - means), 2.0 * var_safe) - 0.5 * np.log(2.0 * np.pi)).sum()


class GMM:
    """Gaussian Mixture Model with diagonal covariance and variance floor."""

    def __init__(self, mfcc_data, n_components, random_state=0):
        self.mfcc_data = mfcc_data
        self.n_components = n_components
        self.means = np.tile(np.mean(self.mfcc_data, axis=0), (n_components, 1))
        rng = np.random.default_rng(random_state)
        data_std = np.sqrt(np.maximum(np.var(self.mfcc_data, axis=0), MIN_VAR))
        for k in range(n_components):
            self.means[k] += 0.01 * rng.standard_normal() * data_std
        self.var = np.tile(np.maximum(np.var(self.mfcc_data, axis=0), MIN_VAR), (n_components, 1))
        self.weights = np.ones(n_components) / n_components

    def _frame_log_probs(self, x):
        """Compute (N, K) matrix of log(weight_k * N(x_i | mu_k, var_k))."""
        var_safe = np.maximum(self.var, MIN_VAR)
        # x: (N, D), means: (K, D), var_safe: (K, D)
        diff = x[:, None, :] - self.means[None, :, :]
        log_gauss = -0.5 * (
            np.log(2.0 * np.pi) + np.log(var_safe[None, :, :]) + (diff ** 2) / var_safe[None, :, :]
        ).sum(axis=2)
        return np.log(np.maximum(self.weights, 1e-12))[None, :] + log_gauss

    def e_step(self, x):
        """Expectation-step returning normalized posterior responsibilities (N, K)."""
        log_weighted = self._frame_log_probs(x)
        log_norm = logsumexp(log_weighted, axis=1, keepdims=True)
        return np.exp(log_weighted - log_norm)

    def m_step(self, x, resp):
        """Maximization-step updating weights, means, and floored variances."""
        nk = np.maximum(np.sum(resp, axis=0), 1e-12)
        self.weights = nk / np.sum(nk)
        self.means = (resp.T @ x) / nk[:, None]
        for k in range(self.n_components):
            diff = x - self.means[k]
            self.var[k] = np.maximum((resp[:, k:k + 1] * (diff ** 2)).sum(axis=0) / nk[k], MIN_VAR)

    def train(self, x):
        resp = self.e_step(x)
        self.m_step(x, resp)

    def log_prob(self, x):
        """Total log-likelihood of sequence x under the GMM."""
        log_weighted = self._frame_log_probs(x)
        return float(np.sum(logsumexp(log_weighted, axis=1)))


def train_model_gmm(train_dir, n_components=6, max_iter=100, random_state=0, cmvn=True):
    """Train one custom GMM per digit 0..9."""
    sequences = get_mfcc_sequences(train_dir, cmvn=cmvn)
    mfcc_data = [np.vstack(sequences[i]) for i in range(10)]
    gmm_models = [GMM(mfcc_data[i], n_components=n_components, random_state=random_state) for i in range(10)]

    lower_bound = -np.inf
    for it in range(1, max_iter + 1):
        prev_lower_bound = lower_bound
        total_log_like = 0.0
        for i in range(10):
            gmm_models[i].train(mfcc_data[i])
            total_log_like += gmm_models[i].log_prob(mfcc_data[i])
        lower_bound = total_log_like
        if abs(lower_bound - prev_lower_bound) < 1.0:
            break
        print(f"Iteration: {it}  log_prob: {lower_bound:.2f}")

    return gmm_models


def predict_from_features(gmm_models, features_mfcc):
    """
    Predict digit from an MFCC feature matrix.
    :return: (predicted_digit_int, margin_over_second_best, scores_array)
    """
    scores = np.array([m.log_prob(features_mfcc) for m in gmm_models])
    sorted_idx = np.argsort(scores)[::-1]
    best = int(sorted_idx[0])
    margin = float(scores[best] - scores[sorted_idx[1]])
    return best, margin, scores


def predict_gmm(gmm_models, test_dir, cmvn=True):
    """Evaluate trained GMM models on test_dir."""
    count = 0
    pred_true = 0
    confusion_matrix = np.zeros((10, 10), dtype=int)

    for test_file, ground_truth in get_test_files(test_dir):
        features_mfcc = mfcc(test_file, cmvn=cmvn)
        pred, _, _ = predict_from_features(gmm_models, features_mfcc)
        if pred == ground_truth:
            pred_true += 1
        confusion_matrix[ground_truth, pred] += 1
        count += 1

    print("---------- Gaussian Mixture Model (GMM) ----------")
    print("confusion_matrix: \n", confusion_matrix)
    print(f"Train num: 160, Test num: {count}, Predict true num: {pred_true}")
    print(f"Accuracy: {pred_true / count:.2f}")


if __name__ == '__main__':
    train_dir = './processed_train_records'
    test_dir = './processed_test_records'
    gmm_models = train_model_gmm(train_dir, n_components=6)
    save_models(gmm_models, './models/gmm_custom.joblib')
    predict_gmm(gmm_models, test_dir)
