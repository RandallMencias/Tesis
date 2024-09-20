import numpy as np
from sklearn.metrics import mutual_info_score

from Metaheuristicas.fitness_functions import load_and_preprocess_data


class AntColonyOptimization:
    def __init__(self, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.n_ants = n_ants  # Number of ants
        self.n_best = n_best  # Number of top solutions to use for pheromone update
        self.n_iterations = n_iterations  # Number of iterations
        self.decay = decay  # Pheromone decay rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic importance

    def _heuristic_information(self, X, y, feature_idx):
        # Use variance as a heuristic for example (can be replaced with other metrics)
        return np.var(X[:, feature_idx])

    def _initialize_pheromone(self, n_features):
        # Initialize pheromone levels (same for all features)
        return np.ones(n_features)

    def fit(self, X, y, fitness_function):
        n_features = X.shape[1]
        pheromone = self._initialize_pheromone(n_features)

        best_solutions = []
        best_fitness_values = []

        for iteration in range(self.n_iterations):
            print(f"Iteration {iteration + 1}/{self.n_iterations}")

            # Ants construct solutions (feature subsets)
            solutions = []
            for ant in range(self.n_ants):
                # Each ant probabilistically selects features
                probabilities = pheromone ** self.alpha * np.array(
                    [self._heuristic_information(X, y, i) for i in range(n_features)]) ** self.beta
                probabilities /= np.sum(probabilities)  # Normalize probabilities

                selected_features = np.random.rand(n_features) < probabilities  # Binary feature selection
                solutions.append(selected_features)

            # Evaluate solutions
            fitness_values = [fitness_function(sol, X, y) for sol in solutions]
            best_ants = np.argsort(fitness_values)[-self.n_best:]  # Top 'n_best' solutions

            # Update pheromones based on best solutions
            for i in range(n_features):
                pheromone[i] *= (1 - self.decay)  # Decay pheromones over time

                # Increase pheromone for good features
                for best_ant in best_ants:
                    if solutions[best_ant][i] == 1:
                        pheromone[i] += 1.0 / (1 + fitness_values[best_ant])

            best_solution_idx = np.argmax(fitness_values)
            best_solutions.append(solutions[best_solution_idx])
            best_fitness_values.append(fitness_values[best_solution_idx])
            print(f"Best fitness of iteration {iteration + 1}: {fitness_values[best_solution_idx]:.4f}")

        # Return best solution across all iterations
        best_overall_idx = np.argmax(best_fitness_values)
        return best_solutions[best_overall_idx], best_fitness_values[best_overall_idx]


# Example usage
def mutual_information_eval(selected_features, data, labels):
    selected_data = data[:, selected_features == 1]
    if selected_data.shape[1] == 0:
        return -np.inf
    return np.sum([mutual_info_score(selected_data[:, i], labels) for i in range(selected_data.shape[1])])

# Assuming X and y are already defined
aco = AntColonyOptimization(n_ants=10, n_best=5, n_iterations=20, decay=0.1, alpha=1, beta=2)
X,y = load_and_preprocess_data()
best_solution, best_fitness = aco.fit(X.values, y.values, fitness_function=mutual_information_eval)

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")