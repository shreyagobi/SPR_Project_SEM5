# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
from scipy.special import logsumexp
from utils import mfcc, get_mfcc_sequences, get_test_files, save_models

warnings.filterwarnings('ignore')

MIN_VAR = 1e-6
EPS = 1e-12


def log_gaussian_prob(x, means, var):
    """
    Estimate the diagonal-covariance log Gaussian probability for frame x.
    """
    var_safe = np.maximum(var, MIN_VAR)
    return (-0.5 * np.log(var_safe) - np.divide(np.square(x - means), 2.0 * var_safe) - 0.5 * np.log(2.0 * np.pi)).sum()


class HMM:
    """Left-to-right / ergodic Hidden Markov Model with single Gaussian emissions."""

    def __init__(self, mfcc_data, n_components):
        self.mfcc_data = mfcc_data
        self.n_components = n_components
        self.init_prob = np.hstack((np.ones(1), np.zeros(n_components - 1)))
        self.means = np.tile(np.mean(self.mfcc_data, axis=0), (n_components, 1))
        self.var = np.tile(np.maximum(np.var(self.mfcc_data, axis=0), MIN_VAR), (n_components, 1))
        self.states = []
        self.trans_mat = np.zeros((self.n_components, self.n_components))

    def _emission_log_probs(self, single_wav):
        """Compute (T, M) log emission probabilities for all frames and states."""
        var_safe = np.maximum(self.var, MIN_VAR)
        diff = single_wav[:, None, :] - self.means[None, :, :]
        return -0.5 * (
            np.log(2.0 * np.pi) + np.log(var_safe[None, :, :]) + (diff ** 2) / var_safe[None, :, :]
        ).sum(axis=2)

    def viterbi(self, data):
        """Update self.states with the most likely state sequence per utterance."""
        log_init = np.log(np.maximum(self.init_prob, EPS))
        log_trans = np.log(np.maximum(self.trans_mat, EPS))
        for index, single_wav in enumerate(data):
            n = single_wav.shape[0]
            emit = self._emission_log_probs(single_wav)
            log_delta = np.zeros((n, self.n_components))
            psi = np.zeros((n, self.n_components), dtype=int)
            log_delta[0] = log_init + emit[0]
            for t in range(1, n):
                scores = log_delta[t - 1][:, None] + log_trans
                psi[t] = np.argmax(scores, axis=0)
                log_delta[t] = np.max(scores, axis=0) + emit[t]
            labeled_seq = np.zeros(n, dtype=int)
            labeled_seq[n - 1] = np.argmax(log_delta[n - 1])
            for t in reversed(range(n - 1)):
                labeled_seq[t] = psi[t + 1, labeled_seq[t + 1]]
            self.states[index] = labeled_seq

    def forward(self, data):
        """Compute forward log-probabilities (T, M) using logsumexp."""
        n = data.shape[0]
        emit = self._emission_log_probs(data)
        log_init = np.log(np.maximum(self.init_prob, EPS))
        log_trans = np.log(np.maximum(self.trans_mat, EPS))
        alpha = np.zeros((n, self.n_components))
        alpha[0] = log_init + emit[0]
        for t in range(1, n):
            alpha[t] = emit[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
        return alpha

    def forward_backward(self, data):
        """Estimate HMM parameters from current state alignments (Segmental K-Means)."""
        self.trans_mat.fill(0.0)
        gamma_0 = np.zeros(self.n_components)
        gamma_1 = np.zeros((self.n_components, data[0].shape[1]))
        gamma_2 = np.zeros((self.n_components, data[0].shape[1]))

        for index, single_wav in enumerate(data):
            n = single_wav.shape[0]
            labeled_seq = self.states[index]
            for t in range(n - 1):
                self.trans_mat[labeled_seq[t], labeled_seq[t + 1]] += 1.0
            for t in range(n):
                s = labeled_seq[t]
                gamma_0[s] += 1.0
                gamma_1[s] += single_wav[t]
                gamma_2[s] += np.square(single_wav[t])

        gamma_0_safe = np.maximum(gamma_0, 1.0)[:, None]
        self.means = gamma_1 / gamma_0_safe
        self.var = np.maximum((gamma_2 - gamma_0_safe * (self.means ** 2)) / gamma_0_safe, MIN_VAR)
        row_sums = self.trans_mat.sum(axis=1, keepdims=True)
        # Avoid division by zero for any unvisited state
        for j in range(self.n_components):
            if row_sums[j, 0] > 0:
                self.trans_mat[j] /= row_sums[j, 0]
            else:
                self.trans_mat[j, j] = 1.0

    def train(self, data, iteration):
        if iteration == 0:
            self.states = []
            for single_wav in data:
                n = single_wav.shape[0]
                init_seq = np.array([min(self.n_components - 1, int(self.n_components * i / n)) for i in range(n)], dtype=int)
                self.states.append(init_seq)
        self.forward_backward(data)
        self.viterbi(data)

    def log_prob(self, data):
        """Return total forward log-likelihood of observation sequence data."""
        alpha = self.forward(data)
        return float(logsumexp(alpha[-1]))


def train_model_hmm(train_dir, n_components=3, max_iter=100, cmvn=True):
    """Train one custom HMM per digit 0..9."""
    sequences = get_mfcc_sequences(train_dir, cmvn=cmvn)
    hmm_models = [HMM(np.vstack(sequences[i]), n_components=n_components) for i in range(10)]

    lower_bound = -np.inf
    for it in range(max_iter + 1):
        prev_lower_bound = lower_bound
        total_log_like = 0.0
        for i in range(10):
            hmm_models[i].train(sequences[i], it)
            for single_wav in sequences[i]:
                total_log_like += hmm_models[i].log_prob(single_wav)
        lower_bound = total_log_like
        if abs(lower_bound - prev_lower_bound) < 0.1:
            break
        print(f"Iteration: {it}  log_prob: {lower_bound:.2f}")

    return hmm_models


def predict_from_features(hmm_models, features_mfcc):
    """
    Predict digit from an MFCC feature matrix.
    :return: (predicted_digit_int, margin_over_second_best, scores_array)
    """
    scores = np.array([m.log_prob(features_mfcc) for m in hmm_models])
    sorted_idx = np.argsort(scores)[::-1]
    best = int(sorted_idx[0])
    margin = float(scores[best] - scores[sorted_idx[1]])
    return best, margin, scores


def predict_hmm(hmm_models, test_dir, cmvn=True):
    """Evaluate trained HMM models on test_dir."""
    count = 0
    pred_true = 0
    confusion_matrix = np.zeros((10, 10), dtype=int)

    for test_file, ground_truth in get_test_files(test_dir):
        features_mfcc = mfcc(test_file, cmvn=cmvn)
        pred, _, _ = predict_from_features(hmm_models, features_mfcc)
        if pred == ground_truth:
            pred_true += 1
        confusion_matrix[ground_truth, pred] += 1
        count += 1

    print("---------- HMM (Hidden Markov Model) ----------")
    print("confusion_matrix: \n", confusion_matrix)
    print(f"Train num: 160, Test num: {count}, Predict true num: {pred_true}")
    print(f"Accuracy: {pred_true / count:.2f}")


if __name__ == '__main__':
    train_dir = './processed_train_records'
    test_dir = './processed_test_records'
    hmm_models = train_model_hmm(train_dir, n_components=3, max_iter=100)
    save_models(hmm_models, './models/hmm_custom.joblib')
    predict_hmm(hmm_models, test_dir)
