import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ACO:
    def __init__(self, n_ants, n_iterations, alpha=1, beta=2, evaporation_rate=0.5, q=1):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta  # Influence of heuristic information
        self.evaporation_rate = evaporation_rate
        self.q = q  # Amount of pheromone deposited by the best ant

    def _initialize_pheromones(self, n_features):
        """Initialize the pheromone matrix."""
        return np.ones(n_features)

    def _heuristic_information(self, X, y, feature_idx):
        """Heuristic information, typically related to feature importance or relevance."""
        return np.var(X[:, feature_idx])  # Variance of the feature as an example

    def _construct_solution(self, pheromones, heuristic_info, n_features):
        """Construct a feature subset based on pheromone and heuristic information."""
        probabilities = np.zeros(n_features)
        for i in range(n_features):
            probabilities[i] = (pheromones[i] ** self.alpha) * (heuristic_info[i] ** self.beta)
        probabilities /= probabilities.sum()
        selected_features = np.random.choice(np.arange(n_features), size=int(n_features / 2), p=probabilities,
                                             replace=False)
        return selected_features

    def _update_pheromones(self, pheromones, ant_solutions, fitness_values, best_solution, best_fitness):
        """Update the pheromones for each feature based on ant solutions and their fitness."""
        pheromones *= (1 - self.evaporation_rate)  # Evaporation
        for i in range(len(ant_solutions)):
            for feature in ant_solutions[i]:
                pheromones[feature] += fitness_values[i] / best_fitness
        for feature in best_solution:
            pheromones[feature] += self.q  # Best solution reinforcement

    def fit(self, X, y, fitness_function):
        """Run the ACO optimization for feature selection."""
        n_features = X.shape[1]
        pheromones = self._initialize_pheromones(n_features)
        best_solution = None
        best_fitness = -np.inf

        for iteration in range(self.n_iterations):
            ant_solutions = []
            fitness_values = []

            # Heuristic information (e.g., feature relevance)
            heuristic_info = np.array([self._heuristic_information(X, y, i) for i in range(n_features)])

            for _ in range(self.n_ants):
                # Construct a solution (feature subset)
                selected_features = self._construct_solution(pheromones, heuristic_info, n_features)
                ant_solutions.append(selected_features)

                # Evaluate the fitness of the feature subset using the given fitness function
                X_selected = X[:, selected_features]
                fitness = fitness_function(X_selected, y)
                fitness_values.append(fitness)

                # Update the best solution found
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = selected_features

            # Update pheromones based on ant solutions
            self._update_pheromones(pheromones, ant_solutions, fitness_values, best_solution, best_fitness)
            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best Fitness: {best_fitness}")

        return best_solution, best_fitness


# Fitness function for feature subset (e.g., accuracy with Random Forest)
def fitness_function(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


# Example usage with a dataset
# if __name__ == "__main__":
#     from sklearn.datasets import load_iris
#
#     data = load_iris()
#     X, y = data.data, data.target
#
#     # ACO parameters
#     aco = ACO(n_ants=10, n_iterations=20)
#     best_solution, best_fitness = aco.fit(X, y, fitness_function)
#
#     print("Best Solution (Selected Features):", best_solution)
#     print("Best Fitness (Accuracy):", best_fitness)
