import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from skrebate import ReliefF


def mutual_information_eval(solution, data, labels):
    # Convert NumPy array back to DataFrame
    data_df = pd.DataFrame(data)
    selected_data = data_df.iloc[:, solution == 1]
    if selected_data.shape[1] == 0:
        return -np.inf
    mi_scores = mutual_info_classif(selected_data, labels)
    return np.sum(mi_scores)


def chi2_eval(solution, data, labels):
    # Convert NumPy array back to DataFrame
    data_df = pd.DataFrame(data)
    selected_data = data_df.iloc[:, solution == 1]
    if selected_data.shape[1] == 0:
        return -np.inf
    chi2_scores, _ = chi2(selected_data, labels)
    return np.mean(chi2_scores)


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def relieff_eval(solution, data, labels, n_neighbors=10):

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    selected_features = data.iloc[:, solution.astype(bool)]

    if selected_features.shape[1] == 0:
        return -np.inf

    labels = np.array(labels)

    relief = ReliefF(n_neighbors=n_neighbors)
    relief.fit(selected_features.values, labels)

    relieff_score = relief.feature_importances_.mean()

    return relieff_score

def load_and_preprocess_data(filename='Resources/SeisBenchV1_v1_1.json'):
    """
    Load and preprocess data from a JSON file.

    Parameters:
    - filename: Path to the JSON file.

    Returns:
    - X_scaled: Scaled feature matrix (DataFrame).
    - y: Target variable.
    """
    with open(filename) as file:
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





