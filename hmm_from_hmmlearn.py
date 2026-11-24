# -*- coding: utf-8 -*-
import numpy as np
from hmmlearn import hmm
from utils import mfcc, get_mfcc_sequences, get_test_files, save_models


def train_model_hmm(train_dir, n_components=4, n_iter=1800, cmvn=True):
    """Train one hmmlearn GaussianHMM per digit 0..9 with proper sequence lengths."""
    sequences = get_mfcc_sequences(train_dir, cmvn=cmvn)
    hmm_models = []
    for digit in range(10):
        seqs = sequences[digit]
        X = np.vstack(seqs)
        lengths = [len(s) for s in seqs]
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type='diag',
            n_iter=n_iter,
            random_state=0,
        )
        np.seterr(all='ignore')
        model.fit(X, lengths=lengths)
        hmm_models.append((model, str(digit)))
    return hmm_models


def predict_from_features(hmm_models, features_mfcc):
    """
    Predict digit from an MFCC feature matrix.
    :return: (predicted_digit_int, margin_over_second_best, scores_array)
    """
    scores = np.array([model.score(features_mfcc) for model, _ in hmm_models])
    sorted_idx = np.argsort(scores)[::-1]
    best = int(sorted_idx[0])
    margin = float(scores[best] - scores[sorted_idx[1]])
    return best, margin, scores


def predict_hmm(hmm_models, test_dir, cmvn=True):
    """Evaluate trained hmmlearn GaussianHMM models on test_dir."""
    count = 0
    pred_true = 0
    for test_file, ground_truth in get_test_files(test_dir):
        features_mfcc = mfcc(test_file, cmvn=cmvn)
        pred, _, _ = predict_from_features(hmm_models, features_mfcc)
        count += 1
        if pred == ground_truth:
            pred_true += 1
    print("---------- HMM (GaussianHMM) ----------")
    print(f"Train num: 160, Test num: {count}, Predict true num: {pred_true}")
    print(f"Accuracy: {pred_true / count:.2f}")


if __name__ == '__main__':
    hmm_models = train_model_hmm("./processed_train_records")
    save_models(hmm_models, "./models/hmm_hmmlearn.joblib")
    predict_hmm(hmm_models, "./processed_test_records")
