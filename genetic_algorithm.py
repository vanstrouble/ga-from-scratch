import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def onemax(chromosome):
    """Calculate the fitness of a single chromosome.
    Args:
        chromosome: A single binary chromosome.
    Returns:
        The number of ones in the chromosome.
    """
    return np.sum(chromosome)


class GA:
    def __init__(
        self, pop_size, str_size, fitness_func, mutation_rate=0.01, crossover_rate=0.7
    ):
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
        self.population = np.random.randint(
            2, size=(pop_size, str_size), dtype=np.int16
        )
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
            np.arange(len(self.population)), size=len(self.population), p=probabilities
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
                offspring[i, crossover_point:] = offspring[i + 1, crossover_point:]
                offspring[i + 1, crossover_point:] = temp
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

    def plot_evolution(self, history, iter_num, problem_name=None):
        """Plot the fitness evolution over generations.
        Args:
            history: A NumPy array with the best fitness for each generation.
            iter_num: The total number of iterations.
            problem_name: Optional problem name to include in the title.
        """
        CORAL = "#DC8665"
        TEAL = "#138086"
        PURPLE = "#534666"
        # ROSE = "#CD7672"
        AMBER = "#EEB462"

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#1C1C2E")
        ax.set_facecolor("#1C1C2E")

        generations = np.arange(1, iter_num + 1)

        # Area under the curve
        ax.fill_between(generations, history, alpha=0.15, color=TEAL)
        # Main line
        ax.plot(generations, history, color=TEAL, linewidth=2, zorder=3)
        # Best fitness marker
        best_gen = np.argmax(history) + 1
        ax.scatter(best_gen, self.best_fitness, color=CORAL, s=60, zorder=4)
        ax.annotate(
            f"best: {self.best_fitness}",
            xy=(float(best_gen), float(self.best_fitness)),
            xytext=(8, -14),
            textcoords="offset points",
            color=CORAL,
            fontsize=9,
        )

        title = "Genetic Algorithm — Fitness Evolution"
        if problem_name:
            title += f"  ·  {problem_name}"
        ax.set_title(title, color="white", fontsize=13, pad=14)
        ax.set_xlabel("Generation", color=AMBER, fontsize=10)
        ax.set_ylabel("Best Fitness", color=AMBER, fontsize=10)

        ax.tick_params(colors="#AAAAAA")
        for spine in ax.spines.values():
            spine.set_edgecolor(PURPLE)

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=PURPLE, linestyle="--", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.show()

    def run(self, iter_num=100, elite_size=1, plot_result=False):
        """Run the genetic algorithm for a fixed number of iterations.
        Args:
            iter_num: Number of generations to execute.
            elite_size: Number of best individuals to keep for the next generation.
            plot_result: Whether to plot the fitness evolution.
        Returns:
            A tuple with the best fitness history and the best solution found.
        """
        history = np.empty(shape=iter_num, dtype=np.int16)
        for it in range(iter_num):
            # Apply the fitness function to the current population
            fitness = np.array(
                [self.fitness_func(chromosome) for chromosome in self.population]
            )

            current_best_fitness = np.max(fitness)
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmax(fitness)].copy()

            history[it] = self.best_fitness

            # Elitism: Keep the best solution from the current generation
            elite_idxs = np.argsort(fitness)[-elite_size:]
            elites = self.population[elite_idxs].copy()

            parents = self._selection(fitness)
            offspring = self._crossover(parents)
            mutated_offspring = self._mutation(offspring)

            worst_idxs = np.argsort(
                np.array(
                    [self.fitness_func(chromosome) for chromosome in mutated_offspring]
                )
            )[:elite_size]
            mutated_offspring[worst_idxs] = elites

            self._replacement(mutated_offspring)

        if plot_result:
            self.plot_evolution(history, iter_num)

        print(f"Best solution: {self.best_solution}")
        print(f"Fitness of the best solution: {self.best_fitness}")

        return history, self.best_solution


if __name__ == "__main__":
    # Parameters for the ONEMAX problem
    POP_SIZE = 100
    STR_SIZE = 50
    MUTATION_RATE = 1 / STR_SIZE  # Heuristic for mutation rate
    CROSSOVER_RATE = 0.8
    ITER_NUM = 100

    ga = GA(
        pop_size=POP_SIZE,
        str_size=STR_SIZE,
        fitness_func=onemax,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
    )
    history, best_solution = ga.run(iter_num=ITER_NUM, elite_size=2)
    ga.plot_evolution(history, ITER_NUM, problem_name="One Max")
