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
    selected_features = data.iloc[:, solution.astype(bool)]
    n_samples, n_features = selected_features.shape

    # Initialize the score array
    scores = np.zeros(n_features)

    # Fit the nearest neighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(selected_features)

    for i in range(n_samples):
        # Find the nearest neighbors
        distances, indices = nn.kneighbors(selected_features.iloc[i, :].values.reshape(1, -1), return_distance=True)
        indices = indices[0][1:]  # Exclude the first neighbor (itself)

        # Calculate the score for each feature
        for j in range(n_features):
            diff_same_class = 0
            diff_diff_class = 0
            for idx in indices:
                if labels[i] == labels[idx]:
                    diff_same_class += np.abs(selected_features.iloc[i, j] - selected_features.iloc[idx, j])
                else:
                    diff_diff_class += np.abs(selected_features.iloc[i, j] - selected_features.iloc[idx, j])
            scores[j] += diff_diff_class - diff_same_class

    # Normalize the scores
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


