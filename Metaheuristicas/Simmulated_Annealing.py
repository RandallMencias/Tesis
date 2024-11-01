from random import random, randint

import numpy as np

from Metaheuristicas.fitness_functions import mutual_information_eval, load_and_preprocess_data, \
    relieff_eval


def simulated_annealing(X, y, initial_temperature=1000, cooling_rate=0.95, max_iter=1000, fitness_function=mutual_information_eval):
    """
    Perform simulated annealing to select the best subset of features using a given fitness function.

    Parameters:
    - X: Feature matrix (Pandas DataFrame).
    - y: Target variable (Pandas Series or numpy array).
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
    current_score = fitness_function(current_solution, X, y)

    # Track the best solution found
    best_solution = current_solution.copy()
    best_score = current_score

    temperature = initial_temperature

    for i in range(max_iter):
        # Generate a new neighboring solution by flipping one feature
        new_solution = current_solution.copy()
        flip_index = randint(0, n_features - 1)
        new_solution[flip_index] = 1 - new_solution[flip_index]  # Flip the feature (0 -> 1 or 1 -> 0)

        # Check if the new solution meets the feature count constraint (10 < features < 60)
        num_selected_features = np.sum(new_solution)
        if num_selected_features < 20 or num_selected_features > 60:
            continue  # Skip this iteration if the constraint is not met

        # Evaluate the new solution's fitness
        new_score = fitness_function(new_solution, X, y)

        # Accept the new solution if it's better, or with some probability if it's worse
        if new_score > current_score:
            current_solution = new_solution
            current_score = new_score
        else:
            # Calculate acceptance probability for worse solutions
            delta = new_score - current_score
            acceptance_probability = np.exp(delta / temperature)
            if random() < acceptance_probability:
                current_solution = new_solution
                current_score = new_score

        # Update the best solution if the current solution is better
        if current_score > best_score:
            best_solution = current_solution.copy()
            best_score = current_score

        # Decrease the temperature according to the cooling rate
        temperature *= cooling_rate

    return best_solution, best_score

# Example usage
def main():
    # Load and preprocess the data
    X, y = load_and_preprocess_data()

    # Define the fitness function to be used
    fitness_function = relieff_eval  # or chi2_eval, relieff_eval

    # Run Simulated Annealing
    best_solution, best_score = simulated_annealing( X, y,initial_temperature=1000, cooling_rate=0.95, max_iter=300, fitness_function=fitness_function)

    # Display the results
    selected_features = X.columns[best_solution.astype(bool)].tolist()
    print(f"Selected Features: {selected_features}")
    print("Best Score:", best_score)
    print("Number of Selected Features:", np.sum(best_solution))

if __name__ == "__main__":
    main()
