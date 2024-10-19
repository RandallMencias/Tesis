import numpy as np
from sklearn.metrics import mutual_info_score
from Metaheuristicas.fitness_functions import mutual_information_eval

import warnings
warnings.filterwarnings("ignore")

class AdvancedBinaryAntColonyOptimization:
    def __init__(self, n_ants, n_best, n_iterations, decay=0.6, alpha=0.5, beta=2, local_search_prob=0.3, min_features=20):
        self.n_ants = n_ants  # Number of ants
        self.n_best = n_best  # Number of top solutions to use for pheromone update
        self.n_iterations = n_iterations  # Number of iterations
        self.decay = decay  # Pheromone decay rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic importance
        self.local_search_prob = local_search_prob  # Probability of performing local search
        self.min_features = min_features  # Minimum number of features to select
        self.feature_selection_log = []  # Log for selected features


    def _heuristic_information(self, X, y, feature_idx):
        """Use mutual information as a heuristic for feature importance."""
        selected_data = X[:, feature_idx]
        return mutual_info_score(selected_data, y)

    def _initialize_pheromone(self, n_features):
        """Initialize pheromone levels (same for all features)."""
        return np.ones(n_features)

    def _local_search(self, solution, X, y, fitness_function):
        """Local search: Flip one random feature in/out of the selected subset to explore neighbors."""
        new_solution = solution.copy()
        flip_idx = np.random.randint(len(solution))
        new_solution[flip_idx] = 1 - new_solution[flip_idx]  # Flip 0 to 1 or 1 to 0
        new_fitness = fitness_function(new_solution, X, y)
        return new_solution, new_fitness

    def _update_pheromone(self, pheromone, solutions, fitness_values):
        """Update pheromone with both global and local updates."""
        n_features = len(pheromone)
        pheromone *= (1 - self.decay)  # Global evaporation

        # Increase pheromone for good features (global update)
        best_ants = np.argsort(fitness_values)[-self.n_best:]  # Top 'n_best' solutions
        for i in range(n_features):
            for best_ant in best_ants:
                if solutions[best_ant][i] == 1:
                    pheromone[i] += 1.0 / (1 + fitness_values[best_ant])

        return pheromone

    def _is_diverse(self, solutions):
        """Check if a new solution is diverse enough from the rest."""
        if len(solutions) == 0:
            return True
        return np.mean([np.sum(solutions[-1] ^ solution) for solution in solutions]) > 0.5


    def fit(self, X, y, fitness_function=mutual_information_eval):
        n_features = X.shape[1]
        pheromone = self._initialize_pheromone(n_features)

        best_global_solution = None
        best_global_fitness = -np.inf

        for iteration in range(self.n_iterations):
            solutions = []
            fitness_values = []
            for ant in range(self.n_ants):
                probabilities = pheromone ** self.alpha * np.array(
                    [self._heuristic_information(X, y, i) for i in range(n_features)]) ** self.beta
                probabilities /= np.sum(probabilities)

                selected_features = np.random.rand(n_features) < probabilities

                while np.sum(selected_features) < self.min_features:
                    selected_features[np.random.randint(n_features)] = 1

                # Check diversity before adding the solution
                if len(solutions) == 0 or self._is_diverse(solutions + [selected_features]):
                    solutions.append(selected_features)

                    fitness = fitness_function(selected_features, X, y)

                    if np.random.rand() < self.local_search_prob:
                        selected_features, fitness = self._local_search(selected_features, X, y, fitness_function)

                    fitness_values.append(fitness)

            # Ensure enough solutions were generated
            while len(solutions) < self.n_ants:
                additional_features = np.random.randint(2, size=n_features)  # Random solution
                if self._is_diverse(solutions + [additional_features]):
                    solutions.append(additional_features)
                    fitness_values.append(fitness_function(additional_features, X, y))

            pheromone = self._update_pheromone(pheromone, solutions, fitness_values)

            best_solution_idx = np.argmax(fitness_values)
            if fitness_values[best_solution_idx] > best_global_fitness:
                best_global_solution = solutions[best_solution_idx]
                best_global_fitness = fitness_values[best_solution_idx]
                # print(f"Iteration {iteration + 1}/{self.n_iterations}: New best solution found with fitness {best_global_fitness}")
            else:
                pass

                # print(f"Iteration {iteration + 1}/{self.n_iterations}: No new best solution found")

            self.feature_selection_log.append(np.sum(solutions, axis=0))

        return best_global_solution, best_global_fitness




# Example usage
# def mutual_information_eval(selected_features, data, labels):
#     selected_data = data[:, selected_features == 1]
#     if selected_data.shape[1] == 0:
#         return -np.inf
#     return np.sum([mutual_info_score(selected_data[:, i], labels) for i in range(selected_data.shape[1])])

# Assuming X and y are already defined
# def main():
#     aco = AdvancedBinaryAntColonyOptimization(n_ants=20, n_best=1, n_iterations=10, decay=0.6, alpha=1, beta=2,
#                                               local_search_prob=0.15)
#     X, y = load_and_preprocess_data()
#     best_solution, best_fitness = aco.fit(X.values, y.values, fitness_function=chi2_eval)
#     # Print results
#     best_features = X.columns[best_solution == 1]
#     print(f"Best Features names: {best_features}")
#     print(f"Best solution length: {len(best_solution)}")
#     print(f"Best fitness: {best_fitness}")
#
# main()
