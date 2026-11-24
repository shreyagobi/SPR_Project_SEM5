# -*- coding: utf-8 -*-
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import mfcc, get_mfcc_sequences, get_test_files, save_models


def train_model_gmm(train_dir, n_components=2, cmvn=True):
    """Train one sklearn GaussianMixture per digit 0..9."""
    sequences = get_mfcc_sequences(train_dir, cmvn=cmvn)
    gmm_models = []
    for digit in range(10):
        X = np.vstack(sequences[digit])
        model = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0)
        np.seterr(all='ignore')
        model.fit(X)
        gmm_models.append((model, str(digit)))
    return gmm_models


def predict_from_features(gmm_models, features_mfcc):
    """
    Predict digit from an MFCC feature matrix.
    :return: (predicted_digit_int, margin_over_second_best, scores_array)
    """
    scores = np.array([model.score(features_mfcc) for model, _ in gmm_models])
    sorted_idx = np.argsort(scores)[::-1]
    best = int(sorted_idx[0])
    margin = float(scores[best] - scores[sorted_idx[1]])
    return best, margin, scores


def predict_gmm(gmm_models, test_dir, cmvn=True):
    """Evaluate trained sklearn GMM models on test_dir."""
    count = 0
    pred_true = 0
    for test_file, ground_truth in get_test_files(test_dir):
        features_mfcc = mfcc(test_file, cmvn=cmvn)
        pred, _, _ = predict_from_features(gmm_models, features_mfcc)
        count += 1
        if pred == ground_truth:
            pred_true += 1
    print("---------- GMM (GaussianMixture) ----------")
    print(f"Train num: 160, Test num: {count}, Predict true num: {pred_true}")
    print(f"Accuracy: {pred_true / count:.2f}")


if __name__ == '__main__':
    gmm_models = train_model_gmm("./processed_train_records")
    save_models(gmm_models, "./models/gmm_sklearn.joblib")
    predict_gmm(gmm_models, "./processed_test_records")
