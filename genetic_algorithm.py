import numpy as np
import matplotlib.pyplot as plt


class GA():
    def __init__(self, pop_size, str_size):
        # initialization
        self.population = np.random.randint(2, size=(pop_size, str_size), dtype=np.int16)
        self.best = 0
        self.idx_best_solution = None

    def run(self, iter_num=100):
        history = np.empty(shape=iter_num, dtype=np.int16)
        for it in range(iter_num):
            # Selection
            total = 0
            pop_norm = []
            for chromosome in self.population:
                val = self.fitness_func(chromosome)
                total += val
                pop_norm.append(val)

            pop_norm = tuple(val/total for val in pop_norm)

            selection = []
            for _ in range(self.population.shape[0]):
                rand_num = np.random.rand()
                summation = 0
                for i, val in enumerate(pop_norm):
                    if rand_num >= summation and rand_num < summation + val:
                        selection.append(self.population[i])
                        break
                    else:
                        summation += val
            selection = np.array(selection, dtype=np.int16)

            # Crossing
            not_even = 0 if len(self.population) % 2 == 0 else 1
            for i in range(0, selection.shape[0]-not_even, 2):
                c_point = np.random.randint(1, selection.shape[1])
                temp_arr = selection[i][c_point::].copy()
                selection[i][c_point::] = selection[i+1][c_point::]
                selection[i+1][c_point::] = temp_arr

            # Mutation
            for chromosome in selection:
                idx = np.random.randint(selection.shape[1])
                chromosome[idx] = 0 if chromosome[idx] == 1 else 1

            # Evolution and replacement
            for i in range(self.population.shape[0]):
                if self.fitness_func(selection[i]) > self.fitness_func(self.population[i]):
                    self.population[i] = selection[i]

            # Save the best in history
            for i in range(self.population.shape[0]):
                fit_val = self.fitness_func(self.population[i])
                if fit_val > self.best:
                    self.best = fit_val
                    self.idx_best_solution = i
            history[it] = self.best

        # Graph history
        plt.plot(np.arange(1,iter_num+1), history)
        plt.title('Evolución hasta converger')
        plt.xlabel('Épocas')
        plt.ylabel('Aptitud')
        plt.show()

        return history, self.population[self.idx_best_solution]

    def fitness_func(self, chromosome):
        return sum(bit for bit in chromosome)


if __name__ == '__main__':
    GA(91, 20).run()