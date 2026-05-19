# Genetic Algorithm and SAT Optimization

This repository contains a project developed for the Artificial Intelligence course at the University of Guanajuato. The original academic exercise has been remastered and polished to improve the code quality, clarify the implementation, and extend its use beyond the initial example. The project is centered on the study of genetic algorithms and their application to combinatorial optimization.

## Genetic Algorithm

A genetic algorithm is a population-based search and optimization method inspired by biological evolution. It belongs to the family of evolutionary algorithms and is commonly used when the solution space is large, discontinuous, or difficult to explore with deterministic methods.

The conceptual foundations of genetic algorithms are usually associated with John Holland, whose work in the 1970s formalized the method in a computational setting. Later, researchers such as David E. Goldberg contributed to its dissemination and practical adoption in engineering and optimization problems.

In the academic material included in this repository, especially the [Genetic Algorithm UG AI UDA presentation](UG_UDA_IA_OPT.pdf) and the [Genetic Algorithm UG AI UDA pseudocode](ga_pseudocode.pdf), the method is presented as an iterative process with the following main stages:

1. Initialize a population of candidate solutions.
2. Evaluate each individual with a fitness function.
3. Select the most promising individuals.
4. Apply crossover to combine genetic material.
5. Apply mutation to preserve diversity and avoid premature convergence.
6. Repeat the process until a stopping criterion is reached.

In practice, the implementation in [genetic_algorithm.py](genetic_algorithm.py) follows this structure with selection, crossover, mutation, replacement, and elitism. The example provided in the repository uses the OneMax problem as a reference case, which makes the behavior of the algorithm easier to observe and analyze.

## SAT Problem

The SAT problem, or Boolean satisfiability problem, asks whether there exists an assignment of truth values to variables that satisfies a given logical formula. It is one of the most relevant problems in theoretical computer science and computational complexity, since it was the first problem proven NP-complete.

Historically, SAT became a central benchmark for optimization and reasoning methods because of its broad applicability in verification, planning, scheduling, and formal logic. In this repository, the problem is treated from the same evolutionary perspective used for the genetic algorithm, but now the fitness function measures how many clauses of a conjunctive normal form instance are satisfied.

The SAT implementation in [sat_problem.py](sat_problem.py) reads CNF instances from the [data](data) directory, builds a fitness function from the clauses, and then executes the same genetic algorithm framework to search for satisfying assignments. This makes it possible to compare the optimization behavior of the algorithm across different SAT instances and to study how evolutionary search performs on a classic NP-complete problem.

## Repository References

- [UG_UDA_IA_OPT.pdf](UG_UDA_IA_OPT.pdf): academic presentation used as a conceptual reference for the genetic algorithm section.
- [ga_pseudocode.pdf](ga_pseudocode.pdf): pseudocode reference for the evolutionary workflow.
- [ASTESJ_020416.pdf](ASTESJ_020416.pdf): reference document related to the SAT problem.
- [genetic_algorithm.py](genetic_algorithm.py): core genetic algorithm implementation.
- [sat_problem.py](sat_problem.py): SAT problem model built on top of the genetic algorithm.
- [data](data): CNF benchmark instances used in the SAT experiments.
