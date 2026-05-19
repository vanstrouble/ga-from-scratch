import os
import random
import numpy as np
import genetic_algorithm as GA


def get_random_filename(path="data/"):
    files = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
    ]

    if not files:
        raise FileNotFoundError(f"No files found in '{path}'")

    return random.choice(files)


def read_sat_instance(filename):
    chromosome_length = None
    n_clauses = None
    clause_length = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("c"):
                if "clause length = " in line:
                    clause_length = int(line.split("=")[1].strip())
                continue
            if line.startswith("p cnf"):
                parts = line.split()
                chromosome_length = int(parts[2])
                n_clauses = int(parts[3])
                break

        if chromosome_length is None or n_clauses is None:
            raise ValueError("Invalid CNF file format: missing 'p cnf' line")

        clauses = np.empty((n_clauses, clause_length), dtype=int)  # type: ignore
        clause_index = 0
        # TODO: Remove print statement after debugging
        print(f"Reading {n_clauses} clauses with length {clause_length}...")
        while clause_index < n_clauses:
            clause_line = f.readline().strip()
            if not clause_line or clause_line.startswith("c"):
                continue
            clause = list(map(int, clause_line.split()[:-1]))
            clauses[clause_index] = clause
            clause_index += 1

    return chromosome_length, n_clauses, clauses


def generate_sat_fitness(clauses):
    idxs = np.abs(clauses) - 1
    positive_mask = clauses > 0

    def fitness_func(chromosome):
        gene_values = chromosome[idxs]
        satisfied_literals = (gene_values == positive_mask)
        satisfied_clauses = np.any(satisfied_literals, axis=1)

        return int(np.sum(satisfied_clauses))

    return fitness_func


if __name__ == "__main__":
    filename = get_random_filename()
    print(f"Selected file: {filename}")
    chromosome_length, n_clauses, clauses = read_sat_instance(filename)
    print(f"Chromosome length: {chromosome_length}")
    print(f"Number of clauses: {n_clauses}")
    print(f"Clauses shape: {len(clauses)} x {len(clauses[0])}")
    print(f"Clauses sample: \n{clauses[:5]}")

    fitness_func = generate_sat_fitness(clauses)

    # Hyperparameters
    POP_SIZE = 200
    MUTATION_RATE = 1 / chromosome_length  # Heuristic for mutation rate
    CROSSOVER_RATE = 0.8
    ITER_NUM = 500
    ELITE_SIZE = 2

    # Run the genetic algorithm
    ga = GA.GA(
        pop_size=POP_SIZE,
        str_size=chromosome_length,
        fitness_func=fitness_func,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE
    )

    history, best_solution = ga.run(iter_num=ITER_NUM, elite_size=ELITE_SIZE)

    # Results
    print(f"Best fitness: {ga.best_fitness} / {n_clauses}")
    print(f"Satisfies: {ga.best_fitness / n_clauses * 100:.1f}%")
    print(f"Best solution: {best_solution}")

    ga.plot_evolution(history, ITER_NUM, problem_name=f"3-SAT Problem ({n_clauses} clauses)")
