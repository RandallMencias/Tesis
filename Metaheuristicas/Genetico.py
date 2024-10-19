import numpy as np
from sklearn.feature_selection import mutual_info_classif
import random
from Metaheuristicas.fitness_functions import mutual_information_eval, load_and_preprocess_data



# Function to calculate mutual information for a subset of features


def genetic_algorithm(X, y, population_size=42, num_parents=28, generations=100, mutation_rate=0.1, crossover_rate=0.8, fitness_function=mutual_information_eval):
    n_features = X.shape[1]

    # Initialize a random population of individuals (feature subsets)
    population = [np.random.choice([0, 1], size=n_features) for _ in range(population_size)]
    # population =population_size
    best_solution = None
    best_fitness = -float('inf')

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        # Evaluate the fitness of each individual in the population
        fitness_scores = []
        for individual in population:
            fitness = fitness_function(individual, X, y)
            fitness_scores.append(fitness)

            # Update the best solution found
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual.copy()

        # Selection: Select individuals based on their fitness (roulette wheel selection)
        fitness_sum = sum(fitness_scores)
        if fitness_sum == 0:
            probabilities = [1 / len(fitness_scores)] * len(fitness_scores)
        else:
            probabilities = [fitness / fitness_sum for fitness in fitness_scores]

        selected_population = random.choices(population, weights=probabilities, k=num_parents)

        # Crossover: Create new population using crossover
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i % num_parents]
            parent2 = selected_population[(i + 1) % num_parents]

            if random.random() < crossover_rate:
                # Perform crossover (single-point crossover) while maintaining feature vector length
                crossover_point = random.randint(1, n_features - 1)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            new_population.append(child1)
            new_population.append(child2)

        # Mutation: Mutate the new population without modifying the length
        for individual in new_population:
            for feature in range(n_features):
                if random.random() < mutation_rate:
                    # Flip the bit for feature selection (1 becomes 0, 0 becomes 1)
                    individual[feature] = 1 - individual[feature]  # Flip the feature bit

        # Replace the old population with the new population
        population = new_population

    return best_solution, best_fitness

# if __name__ == "__main__":
#     X, y = load_and_preprocess_data()
#     best_solution, best_fitness = genetic_algorithm(X, y, generations=5)
#
#     # Print solution length
#     print(f"Solution Length: {np.sum(best_solution)}")
#     selected_features = X.columns[best_solution.astype(bool)].tolist()
#     print(f"Selected Features: {selected_features}")
#     print(f"Best Fitness: {best_fitness}")




