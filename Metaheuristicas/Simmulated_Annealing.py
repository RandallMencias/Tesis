import numpy as np
import random
from sklearn.feature_selection import mutual_info_classif
from fitness_functions import mutual_information_eval

# Function to calculate mutual information for a subset of features
def calculate_mutual_info(X, y, selected_features):
    """
    Calculate the total mutual information for the selected feature subset.

    Parameters:
    - X: Feature matrix.
    - y: Target variable.
    - selected_features: List or array of indices representing the selected features.

    Returns:
    - Total mutual information score for the selected features.
    """
    if len(selected_features) == 0:
        return 0  # No features selected, return zero score

    # Subset of X with the selected features
    X_selected = X[:, selected_features]

    # Calculate mutual information between selected features and target
    mi_scores = mutual_info_classif(X_selected, y)

    # Sum the mutual information scores as the fitness
    total_mi = np.sum(mi_scores)

    return total_mi


# Simulated Annealing for Feature Selection
def simulated_annealing(X, y, fitness_function=mutual_information_eval, initial_temperature=1000, cooling_rate=0.95, max_iter=1000):
    """
    Perform simulated annealing to select the best subset of features using a given fitness function.

    Parameters:
    - X: Feature matrix.
    - y: Target variable.
    - fitness_function: Function to evaluate the fitness of a feature subset.
    - initial_temperature: Initial temperature for the annealing process.
    - cooling_rate: Rate at which the temperature decreases.
    - max_iter: Maximum number of iterations.

    Returns:
    - best_solution: Binary array representing the best feature subset found.
    - best_score: The fitness score of the best solution.
    """
    n_features = X.shape[1]

    # Initialize a random solution (initial subset of features)
    current_solution = np.random.choice([0, 1], size=n_features)
    current_score = fitness_function(X, y, np.where(current_solution == 1)[0])

    # Track the best solution found
    best_solution = current_solution.copy()
    best_score = current_score

    temperature = initial_temperature

    for i in range(max_iter):
        # Generate a new neighboring solution by flipping one feature
        new_solution = current_solution.copy()
        flip_index = random.randint(0, n_features - 1)
        new_solution[flip_index] = 1 - new_solution[flip_index]  # Flip the feature (0 -> 1 or 1 -> 0)

        # Evaluate the new solution's fitness
        new_score = fitness_function(X, y, np.where(new_solution == 1)[0])

        # Accept the new solution if it's better, or with some probability if it's worse
        if new_score > current_score:
            current_solution = new_solution
            current_score = new_score
        else:
            # Calculate acceptance probability for worse solutions
            delta = new_score - current_score
            acceptance_probability = np.exp(delta / temperature)
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_score = new_score

        # Update the best solution if the current solution is better
        if current_score > best_score:
            best_solution = current_solution.copy()
            best_score = current_score

        # Decrease the temperature according to the cooling rate
        temperature *= cooling_rate

        # Optional: Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Current Score: {current_score}, Best Score: {best_score}")

    return best_solution, best_score


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Load a sample dataset (Iris dataset)
    data = load_iris()
    X = data.data
    y = data.target

    # Run Simulated Annealing for feature selection
    best_features, best_fitness = simulated_annealing(X, y, initial_temperature=1000, cooling_rate=0.95, max_iter=1000)

    # Output the results
    print("Best feature subset (selected features):", np.where(best_features == 1)[0])
    print("Best fitness (mutual information score):", best_fitness)
