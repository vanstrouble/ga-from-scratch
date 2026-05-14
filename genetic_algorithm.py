import numpy as np
import matplotlib.pyplot as plt


def onemax(chromosome):
    """Calculate the fitness of a single chromosome.
    Args:
        chromosome: A single binary chromosome.
    Returns:
        The number of ones in the chromosome.
    """
    return sum(bit for bit in chromosome)


class GA():
    def __init__(self, pop_size, str_size, fitness_func, mutation_rate=0.01, crossover_rate=0.7):
        """Create a new genetic algorithm instance.
        Args:
            pop_size: Number of individuals in the population.
            str_size: Length of each chromosome.
            fitness_func: The fitness function to use.
            mutation_rate: Probability of mutating each gene.
            crossover_rate: Probability of crossing two parents.
        Returns:
            None.
        """
        self.population = np.random.randint(2, size=(pop_size, str_size), dtype=np.int16)
        self.str_size = str_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_func = fitness_func
        self.best_solution = None
        self.best_fitness = 0

    def _selection(self, fitness):
        """Select the next generation with roulette wheel selection.
        Args:
            fitness: Fitness values for the current population.
        Returns:
            A NumPy array with the selected parents.
        """
        total_fitness = np.sum(fitness)
        if total_fitness == 0:
            probabilities = np.ones(len(self.population)) / len(self.population)
        else:
            probabilities = fitness / total_fitness

        selected_indices = np.random.choice(
            np.arange(len(self.population)),
            size=len(self.population),
            p=probabilities
        )
        return self.population[selected_indices]

    def _crossover(self, parents):
        """Apply single-point crossover to the parent population.
        Args:
            parents: Selected parents for reproduction.
        Returns:
            A NumPy array with the offspring after crossover.
        """
        offspring = np.copy(parents)
        num_parents = len(offspring) if len(offspring) % 2 == 0 else len(offspring) - 1

        for i in range(0, num_parents, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.str_size)

                temp = offspring[i, crossover_point:].copy()
                offspring[i, crossover_point:] = offspring[i+1, crossover_point:]
                offspring[i+1, crossover_point:] = temp
        return offspring

    def _mutation(self, offspring):
        """Mutate the offspring by flipping random bits.
        Args:
            offspring: The chromosomes produced by crossover.
        Returns:
            A NumPy array with the mutated offspring.
        """
        mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate

        offspring[mutation_mask] = 1 - offspring[mutation_mask]
        return offspring

    def _replacement(self, new_population):
        """Replace the current population with a new one.
        Args:
            new_population: The population to store for the next generation.
        Returns:
            None.
        """
        self.population = new_population

    def run(self, iter_num=100):
        """Run the genetic algorithm for a fixed number of iterations.
        Args:
            iter_num: Number of generations to execute.
        Returns:
            A tuple with the best fitness history and the best solution found.
        """
        history = np.empty(shape=iter_num, dtype=np.int16)
        for it in range(iter_num):
            # Apply the fitness function to the current population
            fitness = np.array([self.fitness_func(chromosome) for chromosome in self.population])

            current_best_fitness = np.max(fitness)
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmax(fitness)].copy()

            history[it] = self.best_fitness

            parents = self._selection(fitness)
            offspring = self._crossover(parents)
            mutated_offspring = self._mutation(offspring)
            self._replacement(mutated_offspring)

        plt.plot(np.arange(1, iter_num + 1), history)
        plt.title('Evolución de la Aptitud')
        plt.xlabel('Generaciones')
        plt.ylabel('Mejor Aptitud')
        plt.grid(True)
        plt.show()

        print(f"Mejor solución encontrada: {self.best_solution}")
        print(f"Aptitud de la mejor solución: {self.best_fitness}")

        return history, self.best_solution


if __name__ == '__main__':
    ga = GA(pop_size=200, str_size=50, fitness_func=onemax, mutation_rate=0.70)
    ga.run(iter_num=200)
