import numpy as np
from sklearn.metrics import mutual_info_score
from Metaheuristicas.fitness_functions import load_and_preprocess_data, mutual_information_eval, chi2_eval, relieff_eval


class AdvancedBinaryAntColonyOptimization:
    def __init__(self, n_ants, n_best, n_iterations, decay, alpha=1, beta=1, local_search_prob=0.1):
        self.n_ants = n_ants  # Number of ants
        self.n_best = n_best  # Number of top solutions to use for pheromone update
        self.n_iterations = n_iterations  # Number of iterations
        self.decay = decay  # Pheromone decay rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic importance
        self.local_search_prob = local_search_prob  # Probability of performing local search

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

    def fit(self, X, y, fitness_function=mutual_information_eval):
        """Main optimization loop."""
        n_features = X.shape[1]
        pheromone = self._initialize_pheromone(n_features)

        best_global_solution = None
        best_global_fitness = -np.inf

        for iteration in range(self.n_iterations):
            # Ants construct solutions (feature subsets)
            solutions = []
            fitness_values = []
            for ant in range(self.n_ants):
                # Each ant probabilistically selects features based on pheromone and heuristic information
                probabilities = pheromone ** self.alpha * np.array(
                    [self._heuristic_information(X, y, i) for i in range(n_features)]) ** self.beta
                probabilities /= np.sum(probabilities)  # Normalize probabilities

                selected_features = np.random.rand(n_features) < probabilities  # Binary feature selection
                solutions.append(selected_features)

                # Evaluate solution
                fitness = fitness_function(selected_features, X, y)

                # Perform local search with a probability
                if np.random.rand() < self.local_search_prob:
                    selected_features, fitness = self._local_search(selected_features, X, y, fitness_function)

                fitness_values.append(fitness)

            # Update pheromones based on best solutions
            pheromone = self._update_pheromone(pheromone, solutions, fitness_values)

            # Track the best solution globally
            best_solution_idx = np.argmax(fitness_values)
            if fitness_values[best_solution_idx] > best_global_fitness:
                best_global_solution = solutions[best_solution_idx]
                best_global_fitness = fitness_values[best_solution_idx]

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {best_global_fitness}")

        return best_global_solution, best_global_fitness


# Example usage
# def mutual_information_eval(selected_features, data, labels):
#     selected_data = data[:, selected_features == 1]
#     if selected_data.shape[1] == 0:
#         return -np.inf
#     return np.sum([mutual_info_score(selected_data[:, i], labels) for i in range(selected_data.shape[1])])

# Assuming X and y are already defined
aco = AdvancedBinaryAntColonyOptimization(n_ants=20, n_best=10, n_iterations=20, decay=0.1, alpha=1, beta=2,
                                          local_search_prob=0.1)
X, y = load_and_preprocess_data()
best_solution, best_fitness = aco.fit(X.values, y.values, fitness_function=relieff_eval)

# Print results
best_features = X.columns[best_solution == 1]
print(f"Best Features names: {best_features}")
print(f"Best solution length: {len(best_solution)}")
print(f"Best fitness: {best_fitness}")
