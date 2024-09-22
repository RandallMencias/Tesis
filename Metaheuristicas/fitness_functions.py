import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def mutual_information_eval(solution, data, labels):
    mi_scores = mutual_info_classif(data.iloc[:, solution], labels)
    mi_score = np.mean(mi_scores)
    return mi_score


def chi2_eval(solution, data, labels):
    chi2_scores = chi2(data.iloc[:, solution], labels)
    chi2_score = np.mean(chi2_scores)
    return chi2_score


def relieff_eval(solution, data, labels, n_neighbors=10):
    """
    Evaluate the quality of a feature subset using the ReliefF algorithm.

    Parameters:
    - solution: Binary array indicating the selected features.
    - data: Feature matrix.
    - labels: Target variable.
    - n_neighbors: Number of neighbors to consider.

    Returns:
    - relieff_score: The average ReliefF score for the selected features.
    """
    # Select features based on the solution
    selected_features = data.iloc[:, solution.astype(bool)].to_numpy()
    labels = labels.to_numpy()

    n_samples, n_features = selected_features.shape

    # Fit the nearest neighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(selected_features)

    # Find nearest neighbors for all samples at once
    distances, indices = nn.kneighbors(selected_features)

    # Initialize the score array
    scores = np.zeros(n_features)

    # Efficiently compute differences
    for i in range(n_samples):
        # Neighbors for the current sample (excluding the sample itself)
        neighbors = indices[i, 1:]

        # Boolean mask for same-class and different-class neighbors
        same_class_mask = labels[i] == labels[neighbors]
        diff_class_mask = ~same_class_mask

        # Get the neighbors' features
        current_sample = selected_features[i, :]
        same_class_neighbors = selected_features[neighbors[same_class_mask], :]
        diff_class_neighbors = selected_features[neighbors[diff_class_mask], :]

        # Compute feature differences for both classes
        if same_class_neighbors.shape[0] > 0:
            diff_same_class = np.abs(current_sample - same_class_neighbors).sum(axis=0)
        else:
            diff_same_class = np.zeros(n_features)

        if diff_class_neighbors.shape[0] > 0:
            diff_diff_class = np.abs(current_sample - diff_class_neighbors).sum(axis=0)
        else:
            diff_diff_class = np.zeros(n_features)

        # Update scores
        scores += (diff_diff_class - diff_same_class)

    # Average the scores over the number of samples
    relieff_score = np.mean(scores / n_samples)
    return relieff_score

def load_and_preprocess_data():
    with open('Resources/SeisBenchV1_v1_1.json') as file:
        data = json.load(file)
        data = pd.DataFrame(data)
        data.dropna(inplace=True)
        data.drop(data[data['Type'] == 'REGIONAL'].index, inplace=True)
        data.drop(data[data['Type'] == 'HB'].index, inplace=True)
        data.drop(data[data['Type'] == 'ICEQUAKE'].index, inplace=True)
        data.drop(data[data['Type'] == ''].index, inplace=True)

    label_encoder = LabelEncoder()
    data['Type'] = label_encoder.fit_transform(data['Type'])

    X = data.iloc[:, 1:]
    y = data['Type']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y


