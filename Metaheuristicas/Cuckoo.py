import json

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.special import gamma
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from Metaheuristicas.fitness_functions import mutual_information_eval, relieff_eval, chi2_eval
from Metaheuristicas.fitness_functions import load_and_preprocess_data



# Función para realizar vuelos de Levy
def levy_flight(Lambda=1.5, size=1):
    sigma = (gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / abs(v) ** (1 / Lambda)
    return step


# Búsqueda de cucos con impresiones de seguimiento
def cuckoo_search(n, dim, iter_max, data, labels, pa=0.25, fitness_function=mutual_information_eval):
    # Initialize nests as random binary matrices (n nests, each with 'dim' features)
    nests = np.random.rand(n, dim) > 0.5

    # Calculate the initial fitness of all nests using mutual information
    fitness = np.array([fitness_function(nest.astype(int), data, labels) for nest in nests])

    # Main loop: iterate for 'iter_max' iterations
    for t in range(iter_max):
        # print(f"Iteración {t + 1}/{iter_max}")  # Display current iteration number

        # For each nest, perform the Cuckoo Search process
        for i in range(n):
            # Generate a new solution via Levy flight (binary mutation)
            step_size = levy_flight(size=dim) > 0.5  # Create a random step using Levy flight
            new_nest = np.logical_xor(nests[i], step_size).astype(int)  # Mutate the current nest

            # Evaluate the fitness of the newly generated nest
            new_fitness = fitness_function(new_nest, data, labels)
            # print(f"  Nido {i + 1} fitness: {fitness[i]:.4f} -> {new_fitness:.4f}")
            # Select a random different nest to compare the new solution against
            random_nest_index = np.random.choice([j for j in range(n) if j != i])

            # If the new nest has better fitness than the random nest, replace it
            if new_fitness > fitness[random_nest_index]:
                nests[random_nest_index] = new_nest  # Replace the random nest with the new one
                fitness[random_nest_index] = new_fitness  # Update fitness of the replaced nest
                # print(f"  Nido {random_nest_index + 1} mejorado a fitness {new_fitness:.4f}")

        # Abandon a fraction 'pa' of the worst nests and replace them with new random nests
        n_abandon = int(n * pa)  # Calculate the number of nests to abandon
        worst_nests_indices = np.argsort(fitness)[:n_abandon]  # Indices of the worst nests

        # Replace the worst nests with new random ones and recalculate their fitness
        nests[worst_nests_indices] = np.random.rand(n_abandon, dim) > 0.5
        fitness[worst_nests_indices] = np.array(
            [fitness_function(nest.astype(int), data, labels) for nest in nests[worst_nests_indices]]
        )
        # print(f"  {n_abandon} peores nidos abandonados y reemplazados por nuevos nidos.")

        # Display the best fitness found at the end of this iteration
        best_fitness = np.max(fitness)
        print(f"Mejor fitness al final de la iteración {t + 1}: {best_fitness:.4f}\n")

    # After all iterations, find the best solution (nest) with the highest fitness
    best_idx = np.argmax(fitness)  # Index of the best nest
    best_nest = nests[best_idx]  # The best nest (binary feature selection vector)
    best_fitness = fitness[best_idx]  # The best fitness value
    # print(f"Mejor nido encontrado:\n{best_nest}\nCon un fitness de: {best_fitness:.4f}")

    # Return the final set of nests, their fitness values, and the best solution
    return nests, fitness, best_nest, best_fitness


# Cálculo de importancia per cápita corregido
# def calculate_importance_per_capita(subsets, fitness_scores, data_columns):
#     importance_per_capita = []
#     for i, subset in enumerate(subsets):
#         # Seleccionar características basadas en la solución actual
#         selected_features = [data_columns[j] for j, included in enumerate(subset) if included]
#
#         # Calcular la importancia total del subconjunto basada en las puntuaciones de fitness individuales
#         subset_fitness = fitness_scores[i]
#
#         # Calcular la importancia per cápita (media de importancias)
#         importance = subset_fitness / np.sum(subset)
#         importance_per_capita.append(importance)
#     return importance_per_capita

# def calculate_auc(X, y):
#     y = label_binarize(y, classes=np.unique(y))
#
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     classifier = OneVsRestClassifier(RandomForestClassifier(random_state=42))
#     classifier.fit(X_train, y_train)
#     y_pred_proba = classifier.predict_proba(X_test)
#     auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
#
#     return auc

# Función para el ranking basado en AUC
# def auc_ranking(nests, fitness_scores, X, y):
#     auc_scores = []
#
#     for index, nest in enumerate(nests):
#         selected_features = X.iloc[:, nest.astype(bool)]
#         auc_score = calculate_auc(selected_features, y)
#         auc_scores.append((index, auc_score))
#
#     ranked_subsets = sorted(auc_scores, key=lambda x: x[1], reverse=True)
#     for rank, (index, auc_score) in enumerate(ranked_subsets, start=1):
#         print(f"Rank {rank}: Subconjunto {index + 1} - AUC Score: {auc_score:.4f}")


#
# #Literal A
# filepath = "features_completas.xlsx"
# data = pd.read_excel(filepath)
# print("Data Set Original:")
# data
#
# x_normalized, y = load_and_preprocess_data(filepath)
# print("\nData Set Normalizado:")
# x_normalized
#
# n = 15  #N de nidos`
# dim = 35
# iter_max = 350  #N iteraciones
# # Cuckoo Search
# nests, fitness_scores, best_nest, best_fitness = cuckoo_search(n, dim, iter_max, x_normalized, y)
#
# # Ordenar los nidos por su puntuación de fitness
# sorted_indices = np.argsort(fitness_scores)[::-1]
# nests_sorted = nests[sorted_indices]
# fitness_scores_sorted = fitness_scores[sorted_indices]
#
# # Calcular la importancia per cápita para los 5 mejores subconjuntos
# top_5_nests = nests_sorted[:5]
# top_5_fitness_scores = fitness_scores_sorted[:5]
# importance_per_capita = calculate_importance_per_capita(top_5_nests, top_5_fitness_scores, x_normalized.columns.tolist())
#
# # Imprimir los 5 mejores subconjuntos con su importancia per cápita
# print("\nTop 5 subconjuntos basados en la importancia per cápita:")
# for i in range(5):
#     selected_features_indices = np.where(top_5_nests[i])[0]
#     selected_features = x_normalized.columns[selected_features_indices].tolist()
#     subset_length = len(selected_features)
#     print(f"Subset {i+1}: {selected_features}")
#     print(f"Subset Length {i+1}: {subset_length}")
#     print(f"Importance Per Capita: {importance_per_capita[i]:.9f}")
#     print(f"Fitness: {top_5_fitness_scores[i]:.9f}")
#     print()
#
#
#
#
#
# # Calcular y mostrar la puntuación AUC para los 5 mejores subconjuntos
# # Loop through the top 5 nests
# print("\nAUC Score for the top 5 subsets:")
# for i in range(5):
#     selected_features_indices = np.where(top_5_nests[i])[0]
#     selected_features = x_normalized.columns[selected_features_indices]
#     X_subset = x_normalized[selected_features]
#     auc_score = calculate_auc(X_subset, y)
#     print(f"Subset {i+1} AUC Score: {auc_score:.4f}")





if __name__ == "__main__":
    X, y = load_and_preprocess_data()

    nests, fitness_scores, best_nest, best_fitness = cuckoo_search(10, 84, 3, X, y, fitness_function=relieff_eval)
    sorted_indices = np.argsort(fitness_scores)[::-1]
    nests_sorted = nests[sorted_indices]
    fitness_scores_sorted = fitness_scores[sorted_indices]
#print the best nest features
    print("Best nest features names: ")
    print(X.columns[best_nest.astype(bool)].tolist())

    print("Best fitness score: ")
    print(fitness_scores_sorted[0])


