import abc
from itertools import chain
from multiprocessing.pool import Pool
from typing import List, Optional, Union

import numpy as np

from rl_testing.evolutionary_algorithms.crossovers import Crossover
from rl_testing.evolutionary_algorithms.fitnesses import Fitness
from rl_testing.evolutionary_algorithms.individuals import Individual
from rl_testing.evolutionary_algorithms.mutations import Mutator
from rl_testing.evolutionary_algorithms.selections import Selector
from rl_testing.util.util import FakePool, get_random_state


class Population(abc.ABC):
    def __init__(
        self,
        individuals: List[Individual],
        fitness: Fitness,
        mutator: Mutator,
        crossover: Crossover,
        pool: Optional[Pool] = None,
        _random_state: Optional[int] = None,
    ):
        self.individuals: List[Individual] = individuals
        self.fitness: Fitness = fitness
        self.mutator: Mutator = mutator
        self.crossover: Crossover = crossover
        self.pool = pool if pool is not None else FakePool()
        self.selector: Selector  # Needs to be set by subclass
        self.num_processes = self.pool._processes
        self.random_state = get_random_state(_random_state)

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_individuals(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def evaluate_individuals_async(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def create_next_generation(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def crossover_individuals(self, crossover_prob: float) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_individuals(self, mutation_prob: float) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def best_individual(self) -> Individual:
        raise NotImplementedError

    @abc.abstractmethod
    def worst_individual(self) -> Individual:
        raise NotImplementedError

    @abc.abstractmethod
    def average_fitness(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def unique_individual_fraction(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def size(self) -> int:
        raise NotImplementedError


class SimplePopulation(Population):
    def __init__(
        self,
        individuals: List[Individual],
        fitness: Fitness,
        mutator: Mutator,
        crossover: Crossover,
        selector: Selector,
        pool: Optional[Pool] = None,
        _random_state: Optional[int] = None,
    ):
        super().__init__(individuals, fitness, mutator, crossover, pool, _random_state)
        self.selector = selector

    @property
    def size(self) -> int:
        return len(self.individuals)

    def initialize(self) -> None:
        return

    def evaluate_individuals(self) -> None:
        for individual in self.individuals:
            if individual.fitness is None:
                individual.fitness = self.fitness.evaluate(individual)

    async def evaluate_individuals_async(self) -> None:
        unevaluated_individuals: List[Individual] = []
        for individual in self.individuals:
            if individual.fitness is None:
                unevaluated_individuals.append(individual)

        for individual, fitness_val in zip(
            unevaluated_individuals, await self.fitness.evaluate_async(unevaluated_individuals)
        ):
            individual.fitness = fitness_val

    def create_next_generation(self) -> None:
        population_size = len(self.individuals)
        num_workers = self.num_processes

        # prepare the chunk sizes for the selection
        chunk_sizes = [population_size // num_workers] * num_workers + [
            population_size % num_workers
        ]

        # Select the individuals in parallel
        offspring: List[Individual] = []
        results_async = [
            self.pool.apply_async(
                self.selector,
                # args=(population, chunk_size_i),
                kwds={"individuals": self.individuals, "rounds": chunk_size_i},
            )
            for chunk_size_i in chunk_sizes
        ]
        for result_async in results_async:
            offspring += result_async.get()

        self.individuals = [individual.copy() for individual in offspring]

    def _crossover_individuals(self, crossover_prob: float) -> List[Individual]:
        mated_children: List[Individual] = []

        # Select potential mating candidates
        self.random_state.shuffle(self.individuals)
        couple_candidates = list(zip(self.individuals[::2], self.individuals[1::2]))

        # Filter out the individuals that will mate
        random_values = self.random_state.random(size=len(self.individuals) // 2)
        mating_candidates = [
            couple_candidates[i]
            for i, random_value in enumerate(random_values)
            if random_value < crossover_prob
        ]
        single_children = [
            couple_candidates[i]
            for i, random_value in enumerate(random_values)
            if random_value >= crossover_prob
        ]

        # Apply crossover on the mating candidates
        random_seeds = self.random_state.integers(0, 2**63, len(mating_candidates))
        for individual1, individual2 in chain(
            single_children,
            self.pool.starmap(self.crossover, zip(mating_candidates, random_seeds)),
        ):
            mated_children.append(individual1)
            mated_children.append(individual2)

        return mated_children

    def crossover_individuals(self, crossover_prob: float) -> None:
        mated_children = self._crossover_individuals(crossover_prob)
        self.individuals = mated_children

    def _mutate_individuals(self, mutation_prob: float) -> List[Individual]:
        mutated_children: List[Individual] = []
        random_values = self.random_state.random(size=len(self.individuals))

        # Filter out the individuals that will mutate
        mutation_candidates = [
            self.individuals[i]
            for i, random_value in enumerate(random_values)
            if random_value < mutation_prob
        ]
        non_mutation_candidates = [
            self.individuals[i]
            for i, random_value in enumerate(random_values)
            if random_value >= mutation_prob
        ]

        # Apply mutation on the mutation candidates
        random_seeds = self.random_state.integers(0, 2**63, len(mutation_candidates))
        mutated_children: List[Individual] = []
        for individual in chain(
            self.pool.starmap(self.mutator, zip(mutation_candidates, random_seeds)),
            non_mutation_candidates,
        ):
            mutated_children.append(individual)

        return mutated_children

    def mutate_individuals(self, mutation_prob: float) -> None:
        mutated_children = self._mutate_individuals(mutation_prob)
        self.individuals = mutated_children

    def best_individual(self) -> Individual:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return self.fitness.best_individual(self.individuals)

    def worst_individual(self) -> Individual:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return self.fitness.worst_individual(self.individuals)

    def average_fitness(self) -> float:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return np.mean([individual.fitness for individual in self.individuals])

    def unique_individual_fraction(self) -> float:
        return (
            len(set(self.individuals)) / len(self.individuals)
            if len(self.individuals) > 0
            else 0.0
        )


class DynamicAdaptivePopulation(SimplePopulation):
    def __init__(
        self,
        individuals: List[Individual],
        fitness: Fitness,
        mutator: Mutator,
        crossover: Crossover,
        selector: Selector,
        pool: Optional[Pool] = None,
        _random_state: Optional[int] = None,
    ):
        super().__init__(
            individuals,
            fitness,
            mutator,
            crossover,
            selector,
            pool,
            _random_state,
        )
        self._population_size = len(individuals)

    def initialize(self) -> None:
        self.individuals = self.selector(self.individuals, self._population_size)

    def create_next_generation(self) -> None:
        offspring = self.selector(self.individuals, self._population_size)
        self.individuals = [individual.copy() for individual in offspring]

    def crossover_individuals(self, crossover_prob: float) -> None:
        mated_children = self._crossover_individuals(crossover_prob)
        self.individuals = mated_children[: self._population_size]

    def mutate_individuals(self, mutation_prob: float) -> None:
        mutated_children = self._mutate_individuals(mutation_prob)
        self.individuals = mutated_children[: self._population_size]

    def best_individual(self) -> Individual:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return self.fitness.best_individual(self.individuals)

    def worst_individual(self) -> Individual:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return self.fitness.worst_individual(self.individuals)

    def average_fitness(self) -> float:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return np.mean([individual.fitness for individual in self.individuals])

    def unique_individual_fraction(self) -> float:
        return (
            len(set(self.individuals)) / len(self.individuals)
            if len(self.individuals) > 0
            else 0.0
        )
