import argparse
import asyncio
import logging
import multiprocessing
import time
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chess.engine
import numpy as np
import wandb.sdk
import yaml

import wandb
from rl_testing.config_parsers import get_engine_config
from rl_testing.config_parsers.engine_config_parser import EngineConfig
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.evolutionary_algorithms import (
    CROSSOVER_FUNCTIONS_DICT,
    MUTATION_FUNCTIONS_DICT,
)
from rl_testing.evolutionary_algorithms.crossovers import Crossover
from rl_testing.evolutionary_algorithms.fitness import (
    BoardSimilarityFitness,
    DifferentialTestingFitness,
    EditDistanceFitness,
    Fitness,
    HashFitness,
    PieceNumberFitness,
)
from rl_testing.evolutionary_algorithms.individuals import BoardIndividual
from rl_testing.evolutionary_algorithms.mutations import Mutator
from rl_testing.evolutionary_algorithms.selections import (
    Selector,
    select_tournament_fast,
)
from rl_testing.util.chess import is_really_valid
from rl_testing.util.evolutionary_algorithm import should_decrease_probability
from rl_testing.util.experiment import (
    get_experiment_params_dict,
    store_experiment_params,
)

RESULT_DIR = Path(__file__).parent.parent / Path("results/evolutionary_algorithm")
WANDB_CONFIG_FILE = Path(__file__).parent.parent / Path(
    "configs/evolutionary_algorithm_configs/config_ea_differential_testing.yaml"
)
OPERATOR_PROBABILITIES_FILE = Path(__file__).parent.parent / Path(
    "configs/evolutionary_algorithm_configs/operator_weight_configs/importance_weights.yaml"
)
DEBUG = True
DEBUG_CONFIG = {
    "num_runs_per_config": 6,
    "num_workers": 8,
    "probability_decay": False,
    "early_stopping": True,
    "early_stopping_value": 1.9,
    "num_generations": 50,
    "population_size": 1000,
    "mutation_prob": 1,
    "mutation_strategy": "dynamic",
    "minimum_mutation_probability": 0.01,
    "crossover_prob": 1,
    "crossover_strategy": "dynamic",
    "minimum_crossover_probability": 0.01,
    "tournament_fraction": 0.1,
    "mutate_add_one_piece": True,
    "mutate_castling_rights": True,
    "mutate_flip_board": True,
    "mutate_move_one_piece": True,
    "mutate_move_one_piece_adjacent": True,
    "mutate_move_one_piece_legal": True,
    "mutate_player_to_move": True,
    "mutate_remove_one_piece": True,
    "mutate_rotate_board": True,
    "mutate_substitute_piece": True,
    "crossover_half_board": True,
    "crossover_one_eighth_board": True,
    "crossover_one_quarter_board": True,
    "mutation_probabilities_path": OPERATOR_PROBABILITIES_FILE,
    "crossover_probabilities_path": OPERATOR_PROBABILITIES_FILE,
}

BestFitnessValue = float
WorstFitnessValue = float
AverageFitnessValue = float
UniqueIndividualFraction = float
Time = float


class EvolutionaryAlgorithmConfig:
    def __init__(
        self,
        num_runs_per_config: int,
        num_workers: int,
        probability_decay: bool,
        early_stopping: bool,
        early_stopping_value: float,
        num_generations: int,
        population_size: int,
        mutation_probability: float,
        mutation_strategy: str,
        crossover_probability: float,
        crossover_strategy: str,
        tournament_fraction: float,
        mutation_functions: List[Callable],
        crossover_functions: List[Callable],
        minimum_mutation_probability: Optional[float] = None,
        minimum_crossover_probability: Optional[float] = None,
        mutation_probabilities: Optional[List[float]] = None,
        crossover_probabilities: Optional[List[float]] = None,
    ) -> None:
        self.num_runs_per_config = num_runs_per_config
        self.num_workers = num_workers
        self.probability_decay = probability_decay
        self.early_stopping = early_stopping
        self.early_stopping_value = early_stopping_value
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_prob = mutation_probability
        self.mutation_strategy = mutation_strategy
        self.crossover_prob = crossover_probability
        self.crossover_strategy = crossover_strategy
        self.tournament_fraction = tournament_fraction
        self.mutation_functions = mutation_functions
        self.mutation_probabilities = mutation_probabilities
        self.crossover_functions = crossover_functions
        self.crossover_probabilities = crossover_probabilities
        self.minimum_mutation_probability = minimum_mutation_probability
        self.minimum_crossover_probability = minimum_crossover_probability

        # Check that probability decay is only enabled if mutation_strategy is "all" and if
        # crossover_strategy is "all"
        if self.probability_decay:
            if self.mutation_strategy != "all":
                raise ValueError(
                    "Probability decay can only be enabled if mutation_strategy is 'all'"
                )
            if self.crossover_strategy != "all":
                raise ValueError(
                    "Probability decay can only be enabled if crossover_strategy is 'all'"
                )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EvolutionaryAlgorithmConfig":
        # Given a list of mutation functions, check if they are enabled in the config
        # and if so, add them to the list of mutation functions
        mutation_functions: List[Callable] = []
        for mutation_function in MUTATION_FUNCTIONS_DICT:
            if config.get(mutation_function, False):
                mutation_functions.append(MUTATION_FUNCTIONS_DICT[mutation_function])

        # Check if a path to a file with mutation probabilities is given and if so,
        # load the mutation probabilities from the file
        if config.get("mutation_probabilities_path", None):
            mutation_probabilities = []
            with open(config["mutation_probabilities_path"], "r") as f:
                mutation_probability_dict = yaml.safe_load(f)

            for mutation_function in MUTATION_FUNCTIONS_DICT:
                if (
                    mutation_function in mutation_probability_dict
                    and MUTATION_FUNCTIONS_DICT[mutation_function] in mutation_functions
                ):
                    mutation_probabilities.append(mutation_probability_dict[mutation_function])

            assert len(mutation_probabilities) == len(mutation_functions), (
                "The number of mutation probabilities must be equal to the number of mutation functions."
                f"Got {len(mutation_probabilities)} mutation probabilities and {len(mutation_functions)} mutation functions."
            )
        else:
            mutation_probabilities = None

        # Given a list of crossover functions, check if they are enabled in the config
        # and if so, add them to the list of crossover functions
        crossover_functions: List[Callable] = []
        for crossover_function in CROSSOVER_FUNCTIONS_DICT:
            if config.get(crossover_function, False):
                crossover_functions.append(CROSSOVER_FUNCTIONS_DICT[crossover_function])

        # Check if a path to a file with crossover probabilities is given and if so,
        # load the crossover probabilities from the file
        if config.get("crossover_probabilities_path", None):
            crossover_probabilities = []
            with open(config["crossover_probabilities_path"], "r") as f:
                crossover_probability_dict = yaml.safe_load(f)

            for crossover_function in CROSSOVER_FUNCTIONS_DICT:
                if (
                    crossover_function in crossover_probability_dict
                    and CROSSOVER_FUNCTIONS_DICT[crossover_function] in crossover_functions
                ):
                    crossover_probabilities.append(crossover_probability_dict[crossover_function])

            assert len(crossover_probabilities) == len(crossover_functions), (
                "The number of crossover probabilities must be equal to the number of crossover functions."
                f"Got {len(crossover_probabilities)} crossover probabilities and {len(crossover_functions)} crossover functions."
            )
        else:
            crossover_probabilities = None

        return cls(
            num_runs_per_config=config["num_runs_per_config"],
            num_workers=config["num_workers"],
            probability_decay=config["probability_decay"],
            early_stopping=config["early_stopping"],
            early_stopping_value=config["early_stopping_value"],
            num_generations=config["num_generations"],
            population_size=config["population_size"],
            mutation_probability=config["mutation_prob"],
            mutation_strategy=config["mutation_strategy"],
            crossover_probability=config["crossover_prob"],
            crossover_strategy=config["crossover_strategy"],
            tournament_fraction=config["tournament_fraction"],
            mutation_functions=mutation_functions,
            crossover_functions=crossover_functions,
            minimum_mutation_probability=config.get("minimum_mutation_probability", None),
            minimum_crossover_probability=config.get("minimum_crossover_probability", None),
            mutation_probabilities=mutation_probabilities,
            crossover_probabilities=crossover_probabilities,
        )


def get_random_state(random_state: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Get a random state. Use the provided random state if it is not None, otherwise use the default random state.

    Args:
        random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        np.random.Generator: The random state.
    """
    if random_state is None:
        return np.random.default_rng()
    return random_state


def get_random_individuals(
    file_path: str, amount: int, _random_state: Optional[np.random.Generator]
) -> List[BoardIndividual]:
    """Get a number of random individuals.

    Args:
        file_path (str): The path to the file containing the boards.
        amount (int): The number of boards to get. It must hold that 0 < amount <= 100_000.
        _random_state (Optional[np.random.Generator]): The random state to use. Defaults to None.

    Returns:
        List[BoardIndividual]: The random boards.
    """
    assert 0 < amount <= 100_000, f"Amount must be between 0 and 100_000, got {amount}."

    random_state = get_random_state(_random_state)

    # Read the fen-strings from the provided file.
    with open(file_path, "r") as f:
        lines = f.readlines()

    individuals = []
    chosen_fens = []
    while len(individuals) < amount:
        # Randomly choose 'amount' fen strings from the file
        fens = random_state.choice(lines, size=amount - len(individuals), replace=False)

        # Convert the fen strings to boards
        candidates = [BoardIndividual(fen) for fen in fens]

        # Filter out invalid boards
        candidates = [
            candidate
            for candidate in candidates
            if is_really_valid(candidate) and candidate.fen() not in chosen_fens
        ]
        individuals.extend(candidates)
        chosen_fens.extend([candidate.fen() for candidate in candidates])

    return individuals


def setup_operators(
    random_state: np.random.Generator,
    is_bigger_fitness_better: bool,
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
) -> Tuple[Mutator, Crossover, Selector]:

    # Initialize the mutation operator
    mutate = Mutator(
        mutation_strategy=evolutionary_algorithm_config.mutation_strategy,
        minimum_probability=evolutionary_algorithm_config.minimum_mutation_probability,
        _random_state=random_state,
    )

    # Register the mutation functions
    for mutation_function in evolutionary_algorithm_config.mutation_functions:
        mutate.register_mutation_function(
            [mutation_function],
            retries=5,
            check_game_not_over=True,
            clear_fitness_values=True,
        )

    # Set the mutation probabilities
    if evolutionary_algorithm_config.mutation_probabilities is None:
        mutate_global_prob = 1 / len(evolutionary_algorithm_config.mutation_functions) * 2
        mutate.set_global_probability(mutate_global_prob)
    else:
        for mutation_function, mutation_probability in zip(
            mutate.mutation_functions,
            evolutionary_algorithm_config.mutation_probabilities,
        ):
            mutation_function.probability = mutation_probability

    # Initialize the crossover operator
    crossover = Crossover(
        crossover_strategy=evolutionary_algorithm_config.crossover_strategy,
        minimum_probability=evolutionary_algorithm_config.minimum_crossover_probability,
        _random_state=random_state,
    )

    # Register the crossover functions
    for crossover_function in evolutionary_algorithm_config.crossover_functions:
        crossover.register_crossover_function(
            [crossover_function],
            retries=5,
            check_game_not_over=True,
            clear_fitness_values=True,
        )

    # Set the crossover probabilities
    if evolutionary_algorithm_config.crossover_probabilities is None:
        crossover_global_prob = 1 / len(evolutionary_algorithm_config.crossover_functions) * 2
        crossover.set_global_probability(crossover_global_prob)
    else:
        for crossover_function, crossover_probability in zip(
            crossover.crossover_functions,
            evolutionary_algorithm_config.crossover_probabilities,
        ):
            crossover_function.probability = crossover_probability

    # Initialize the selection operator
    select = Selector(
        selection_strategy="all",
        _random_state=random_state,
    )

    # Register the selection functions
    select.register_selection_function(
        select_tournament_fast,
        tournament_size=int(
            evolutionary_algorithm_config.population_size
            * evolutionary_algorithm_config.tournament_fraction
        ),
        is_bigger_better=is_bigger_fitness_better,
    )

    return mutate, crossover, select


def log_time(start_time: float, message: str = ""):
    """Log the time since the start of the program.

    Args:
        start_time (float): The time the program started.
    """
    end_time = time.time()
    time_elapsed = end_time - start_time
    logging.info(f"Time {message}: {time_elapsed:.2f} seconds.")


def select_individuals(
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
    population: List[BoardIndividual],
    select: Selector,
    pool: multiprocessing.Pool,
) -> List[BoardIndividual]:
    population_size = evolutionary_algorithm_config.population_size
    num_workers = evolutionary_algorithm_config.num_workers

    # prepare the chunk sizes for the selection
    chunk_sizes = [population_size // num_workers] * num_workers + [population_size % num_workers]

    # Select the individuals in parallel
    offspring = []
    results_async = [
        pool.apply_async(
            select,
            # args=(population, chunk_size_i),
            kwds={"individuals": population, "rounds": chunk_size_i},
        )
        for chunk_size_i in chunk_sizes
    ]
    for result_async in results_async:
        offspring += result_async.get()

    return offspring


def recombine_individuals(
    population: List[BoardIndividual],
    crossover: Crossover,
    pool: multiprocessing.Pool,
    chunk_size: int,
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
    random_state: np.random.Generator,
) -> List[BoardIndividual]:
    mated_children: List[BoardIndividual] = []
    couple_candidates = list(zip(population[::2], population[1::2]))
    random_values = random_state.random(size=len(population) // 2)

    # Filter out the individuals that will mate
    mating_candidates = [
        couple_candidates[i]
        for i, random_value in enumerate(random_values)
        if random_value < evolutionary_algorithm_config.crossover_prob
    ]
    single_children = [
        couple_candidates[i]
        for i, random_value in enumerate(random_values)
        if random_value >= evolutionary_algorithm_config.crossover_prob
    ]

    # Apply crossover on the mating candidates
    random_seeds = random_state.integers(0, 2**63, len(mating_candidates))
    for individual1, individual2 in chain(
        single_children, pool.starmap(crossover, zip(mating_candidates, random_seeds))
    ):
        mated_children.append(individual1)
        mated_children.append(individual2)

    return mated_children


def mutate_individuals(
    population: List[BoardIndividual],
    mutate: Mutator,
    pool: multiprocessing.Pool,
    chunk_size: int,
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
    random_state: np.random.Generator,
) -> List[BoardIndividual]:
    mutated_children: List[BoardIndividual] = []
    random_values = random_state.random(size=len(population))

    # Filter out the individuals that will mutate
    mutation_candidates = [
        population[i]
        for i, random_value in enumerate(random_values)
        if random_value < evolutionary_algorithm_config.mutation_prob
    ]
    non_mutation_candidates = [
        population[i]
        for i, random_value in enumerate(random_values)
        if random_value >= evolutionary_algorithm_config.mutation_prob
    ]

    # Apply mutation on the mutation candidates
    random_seeds = random_state.integers(0, 2**63, len(mutation_candidates))
    mutated_children: List[BoardIndividual] = []
    for individual in chain(
        pool.starmap(mutate, zip(mutation_candidates, random_seeds)),
        non_mutation_candidates,
    ):
        mutated_children.append(individual)

    return mutated_children


async def evolutionary_algorithm(
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
    engine_generator1: EngineGenerator,
    engine_generator2: EngineGenerator,
    search_limits1: Dict[str, Any],
    search_limits2: Dict[str, Any],
    network_name1: Optional[str] = None,
    network_name2: Optional[str] = None,
    num_engines1: int = 1,
    num_engines2: int = 1,
    logger: Optional[logging.Logger] = None,
    seed: Optional[int] = None,
) -> Tuple[
    Time,
    List[BoardIndividual],
    List[BestFitnessValue],
    List[AverageFitnessValue],
    List[WorstFitnessValue],
    List[UniqueIndividualFraction],
]:
    """Run a simple evolutionary algorithm with multiple mutation and crossover functions using differential testing on
    chess engines as fitness values. The function uses python asyncio to run multiple chess engines in separate processes
    locally or on a cluster. Furthermore, the function uses the multiprocessing module to run multiple workers in parallel
    to speed up the mutation and crossover functions.

    Args:
        engine_generator1 (EngineGenerator): A generator for the first chess engine.
        engine_generator2 (EngineGenerator): A generator for the second chess engine.
        search_limits1 (Dict[str, Any]): The search limits which the first engine uses for the analysis.
        search_limits2 (Dict[str, Any]): The search limits which the second engine uses for the analysis.
        network_name1 (Optional[str], optional): The name of the network which the first engine uses. Defaults to None.
        network_name2 (Optional[str], optional): The name of the network which the second engine uses. Defaults to None.
        num_engines1 (int, optional): How many engines of type 1 should be run in parallel to speed up the analysis.
        num_engines2 (int, optional): How many engines of type 2 should be run in parallel to speed up the analysis.
        logger (Optional[logging.Logger], optional): A logger which can be used to log the progress of the algorithm.
        num_workers (Optional[int], optional): How many workers should be used to run the mutations and crossovers in
            parallel. Defaults to None which means that all available cores are used.
        seed (Optional[int], optional): The seed for the random number generator. Defaults to None

    Returns:
        Tuple[
            Time,
            List[BoardIndividual],
            List[BestFitnessValue],
            List[AverageFitnessValue],
            List[WorstFitnessValue],
            List[UniqueIndividualFraction],
        ]: The runtime in seconds, the best individual of each generation, the best fitness value of each generation,
            the average fitness value of each generation, the worst fitness value of each generation and the fraction
            of unique individuals in each generation.
    """
    start_time = time.time()
    for parameter in [
        evolutionary_algorithm_config.crossover_prob,
        evolutionary_algorithm_config.mutation_prob,
        evolutionary_algorithm_config.tournament_fraction,
    ]:
        assert 0 <= parameter <= 1, f"Parameter {parameter = } must be between 0 and 1."

    if evolutionary_algorithm_config.num_workers is None:
        evolutionary_algorithm_config.num_workers = multiprocessing.cpu_count()

    random_state = np.random.default_rng(seed)

    # Initialize the operators
    fitness = DifferentialTestingFitness(
        engine_generator1=engine_generator1,
        engine_generator2=engine_generator2,
        search_limits1=search_limits1,
        search_limits2=search_limits2,
        network_name1=network_name1,
        network_name2=network_name2,
        num_engines1=num_engines1,
        num_engines2=num_engines2,
        logger=logger,
    )
    await fitness.create_tasks()

    mutate, crossover, select = setup_operators(
        random_state,
        is_bigger_fitness_better=fitness.is_bigger_better,
        evolutionary_algorithm_config=evolutionary_algorithm_config,
    )

    with multiprocessing.Pool(processes=evolutionary_algorithm_config.num_workers) as pool:
        best_individuals = []
        best_fitness_values = []
        average_fitness_values = []
        worst_fitness_values = []
        unique_individual_fractions = []
        chunk_size = (
            evolutionary_algorithm_config.population_size
            // evolutionary_algorithm_config.num_workers
            // 5
        )

        # Create the population
        log_time(start_time, "before creating population")
        population = get_random_individuals(
            "data/random_positions.txt",
            evolutionary_algorithm_config.population_size,
            random_state,
        )
        log_time(start_time, "after creating population")

        # Evaluate the entire population
        for individual, fitness_val in zip(population, await fitness.evaluate_async(population)):
            individual.fitness = fitness_val

        for generation in range(evolutionary_algorithm_config.num_generations):
            logging.info(f"\n\nGeneration {generation}")

            # Select the next generation individuals
            log_time(start_time, "before selecting")
            offspring = select_individuals(
                evolutionary_algorithm_config=evolutionary_algorithm_config,
                population=population,
                select=select,
                pool=pool,
            )

            # Clone the selected individuals
            log_time(start_time, "before cloning")
            offspring: List[BoardIndividual] = list(map(BoardIndividual.copy, offspring))
            random_state.shuffle(offspring)
            log_time(start_time, "after cloning")

            # Apply crossover on the offspring
            log_time(start_time, "before mating")
            mated_children = recombine_individuals(
                population=offspring,
                crossover=crossover,
                pool=pool,
                chunk_size=chunk_size,
                evolutionary_algorithm_config=evolutionary_algorithm_config,
                random_state=random_state,
            )
            log_time(start_time, "after mating")

            # Apply mutation on the offspring
            log_time(start_time, "before mutating")
            population_to_mutate = (
                offspring if mutate.should_mutate_original_population() else mated_children
            )
            mutated_children = mutate_individuals(
                population=population_to_mutate,
                mutate=mutate,
                pool=pool,
                chunk_size=chunk_size,
                evolutionary_algorithm_config=evolutionary_algorithm_config,
                random_state=random_state,
            )
            log_time(start_time, "after mutating")

            # Evaluate the individuals with an invalid fitness
            log_time(start_time, "before evaluating")
            unevaluated_individuals: List[BoardIndividual] = []
            adapted_population = (
                mated_children + mutated_children
                if mutate.should_mutate_original_population()
                else mutated_children
            )
            for individual in adapted_population:
                if individual.fitness is None:
                    unevaluated_individuals.append(individual)

            for individual, fitness_val in zip(
                unevaluated_individuals, await fitness.evaluate_async(unevaluated_individuals)
            ):
                individual.fitness = fitness_val
            log_time(start_time, "after evaluating")

            # The population is entirely replaced by the offspring
            population = adapted_population

            # Optionally update the crossover and mutation probabilities
            # using data from the previous generation
            mutate.analyze_mutation_effects(population, print_update=True)
            crossover.analyze_crossover_effects(population, print_update=True)

            log_time(start_time, "before finding best individual")
            # Print the best individual and its fitness
            best_individual, best_fitness = fitness.best_individual(population)
            best_individuals.append(best_individual.copy())
            best_fitness_values.append(best_fitness)
            logging.info(f"{best_individual = }, {best_fitness = }")

            log_time(start_time, "before finding worst individual")
            # Print the worst individual and its fitness
            worst_individual, worst_fitness = fitness.worst_individual(population)
            worst_fitness_values.append(worst_fitness)
            logging.info(f"{worst_individual = }, {worst_fitness = }")

            log_time(start_time, "before finding average fitness")
            # Print the average fitness
            average_fitness = (
                sum(individual.fitness for individual in population)
                / evolutionary_algorithm_config.population_size
            )
            average_fitness_values.append(average_fitness)
            logging.info(f"{average_fitness = }")

            log_time(start_time, "before finding unique individuals")
            # Print the number of unique individuals
            unique_individuals = set([p.fen() for p in population])
            unique_individual_fraction = (
                len(unique_individuals) / evolutionary_algorithm_config.population_size
            )
            unique_individual_fractions.append(unique_individual_fraction)
            logging.info(f"Number of unique individuals = {len(unique_individuals)}")

            # Print the adaption history of the best individual
            logging.info(f"{best_individual.history = }")

            # Check if the best fitness is above the early stopping threshold
            if (
                evolutionary_algorithm_config.early_stopping
                and best_fitness >= evolutionary_algorithm_config.early_stopping_value
            ):
                num_generations_remaining = (
                    evolutionary_algorithm_config.num_generations - generation - 1
                )

                # Fill up the lists with the last value
                best_individuals.extend([best_individual.copy()] * (num_generations_remaining))
                best_fitness_values.extend([best_fitness] * (num_generations_remaining))
                average_fitness_values.extend([average_fitness] * (num_generations_remaining))
                worst_fitness_values.extend([worst_fitness] * (num_generations_remaining))
                unique_individual_fractions.extend(
                    [unique_individual_fraction] * (num_generations_remaining)
                )

                logging.info("Early stopping!")
                break

            # Check if the probabilities of mutation and crossover are too high
            if evolutionary_algorithm_config.probability_decay and should_decrease_probability(
                best_fitness, difference_threshold=0.5
            ):
                assert (
                    mutate.global_probability is not None
                ), "Global mutation probability must not be None"
                assert (
                    crossover.global_probability is not None
                ), "Global crossover probability must not be None"
                if mutate.global_probability > 1 / len(mutate.mutation_functions):
                    mutate.set_global_probability(mutate.global_probability / 2)
                    logging.info(f"{mutate.global_probability = }")
                if crossover.global_probability > 1 / len(crossover.crossover_functions):
                    crossover.set_global_probability(crossover.global_probability / 2)
                    logging.info(f"{crossover.global_probability = }")

    end_time = time.time()
    logging.info(f"Number of evaluations: {fitness.num_evaluations}")
    logging.info(f"Total time: {end_time - start_time} seconds")

    # Cancel all running subprocesses which the fitness evaluator spawned
    fitness.cancel_tasks()

    # Log the best individual and its fitness
    best_individual, best_fitness = fitness.best_individual(best_individuals)
    logging.info(
        f"FINAL best individual: {best_individual.fen()}, FINAL best fitness: {best_fitness}"
    )

    # Log the histories of the mutation- and crossover probability distributions
    mutate.print_mutation_probability_history()
    crossover.print_crossover_probability_history()

    return (
        end_time - start_time,
        best_individuals,
        best_fitness_values,
        average_fitness_values,
        worst_fitness_values,
        unique_individual_fractions,
    )


async def run_evolutionary_algorithm_n_times(
    number_of_runs: int,
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
    engine_config1: EngineConfig,
    engine_config2: EngineConfig,
    command_line_args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[
    List[Time],
    List[List[BoardIndividual]],
    List[List[BestFitnessValue]],
    List[List[AverageFitnessValue]],
    List[List[WorstFitnessValue]],
    List[List[UniqueIndividualFraction]],
]:
    """A wrapper function which first initializes the engine generators and then runs the evolutionary algorithm
    `number_of_runs` times. The results of each run are stored in a list and returned.

    Args:
        number_of_runs (int): The number of times the evolutionary algorithm should be run.
        evolutionary_algorithm_config (EvolutionaryAlgorithmConfig): A configuration object for the evolutionary algorithm.
        engine_config1 (EngineConfig): A configuration object for the first engine.
        engine_config2 (EngineConfig): A configuration object for the second engine.
        command_line_args (argparse.Namespace): The command line arguments containing further configuration parameters
            for the engines.
        logger (logging.Logger): A logger object.

    Returns:
        Tuple[
            List[Time],
            List[List[BoardIndividual]],
            List[List[BestFitnessValue]],
            List[List[AverageFitnessValue]],
            List[List[WorstFitnessValue]],
            List[List[UniqueIndividualFraction]],
        ]: A tuple containing a list of run-times of each iteration of the algorithm, a list of lists of
            the best individuals, the best fitness values, the average fitness values, the worst fitness values and the
            unique individual fractions for each generation.
    """

    # Create the engine generators. This must be done inside this function because the ssh connection is run using asyncio, and all
    # asyncio objects need to be created inside the same event loop.
    engine_generator1, engine_generator2 = [
        get_engine_generator(engine_config) for engine_config in [engine_config1, engine_config2]
    ]

    # If the two engine configs are the same, we can use the same engine generator
    if engine_config1 == engine_config2:
        engine_generator2 = engine_generator1

    # Run 'number_of_runs' many runs of the evolutionary algorithm
    result_tuples = []
    for seed_offset in range(number_of_runs):
        logger.info(f"\n\nStarting run {seed_offset + 1}/{number_of_runs}")
        result_tuples.append(
            await evolutionary_algorithm(
                evolutionary_algorithm_config=evolutionary_algorithm_config,
                engine_generator1=engine_generator1,
                engine_generator2=engine_generator2,
                search_limits1=engine_config1.search_limits,
                search_limits2=engine_config2.search_limits,
                network_name1=command_line_args.network_path1,
                network_name2=command_line_args.network_path2,
                num_engines1=command_line_args.num_engines1,
                num_engines2=command_line_args.num_engines2,
                logger=logger,
                seed=command_line_args.seed + seed_offset,
            )
        )

    # Transpose the list of tuples so that the different result lists are in the same tuple
    # This is the same as 2d matrix transposition
    result_tuples = list(map(list, zip(*result_tuples)))

    return result_tuples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # NETWORKS:
    # =========
    # strong and recent: "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    # from leela paper: "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    # strong recommended: "network_600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2" # noqa: E501
    # Weak local 1: "f21ee51844a7548c004a1689eacd8b4cd4c6150d6e03c732b211cf9963d076e1"
    # Weak local 2: "fbd5e1c049d5a46c098f0f7f12e79e3fb82a7a6cd1c9d1d0894d0aae2865826f"

    # fmt: off
    # Engine parameters
    parser.add_argument("--seed",                type=int,  default=42)
    parser.add_argument("--engine_config_name1", type=str,  default="local_400_nodes.ini")  # noqa: E501
    parser.add_argument("--engine_config_name2", type=str,  default="local_400_nodes.ini")  # noqa: E501
    # parser.add_argument("--engine_config_name1", type=str,  default="remote_400_nodes.ini")  # noqa: E501
    # parser.add_argument("--engine_config_name2", type=str,  default="remote_400_nodes.ini")  # noqa: E501
    parser.add_argument("--network_path1",       type=str,  default="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207")  # noqa: E501
    parser.add_argument("--network_path2",       type=str,  default="T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2")  # noqa: E501
    parser.add_argument("--num_engines1" ,       type=int,  default=2)
    parser.add_argument("--num_engines2" ,       type=int,  default=2)
    parser.add_argument("--result_subdir",       type=str,  default="main_results")
    # fmt: on

    ##################################
    #           CONFIG END           #
    ##################################

    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()

    # Parse the arguments
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Get the engine config and engine generator
    engine_config1, engine_config2 = [
        get_engine_config(
            config_name=name,
            config_folder_path=Path(__file__).parent.parent.absolute()
            / Path("configs/engine_configs/"),
        )
        for name in [args.engine_config_name1, args.engine_config_name2]
    ]

    # Create results-file-name
    engine_config_name1 = args.engine_config_name1[:-4]
    engine_config_name2 = args.engine_config_name2[:-4]
    network_name1 = args.network_path1.split("-")[0]
    network_name2 = args.network_path2.split("-")[0]

    if engine_config_name1 == engine_config_name2:
        engine_config_name = engine_config_name1
    else:
        engine_config_name = f"{engine_config_name1}_AND_{engine_config_name2}"

    # Store the experiment config in the results file
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_ENGINES_{engine_config_name}_NETWORKS_{network_name1}_AND_{network_name2}.txt"  # noqa: E501
    )
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Get the experiment config as a dictionary
    experiment_config = get_experiment_params_dict(namespace=args, source_file_path=__file__)

    # Log the experiment config
    experiment_config_str = "{\n"
    for key, value in experiment_config.items():
        experiment_config_str += f"    {key}: {value},\n"
    experiment_config_str += "\n}"

    logger.info(f"\nExperiment config:\n{experiment_config_str}")

    # Read the weights and biases config file
    with open(WANDB_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    if not DEBUG:
        run = wandb.init(config=config, project="rl-testing")
        evolutionary_algorithm_config = EvolutionaryAlgorithmConfig.from_dict(wandb.config)
    else:
        evolutionary_algorithm_config = EvolutionaryAlgorithmConfig.from_dict(DEBUG_CONFIG)

    # Run the evolutionary algorithm 'num_runs_per_config' times
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    (
        run_times,
        populations,
        best_fitness_value_series,
        average_fitness_value_series,
        worst_fitness_value_series,
        unique_individual_fractions,
    ) = asyncio.run(
        run_evolutionary_algorithm_n_times(
            number_of_runs=evolutionary_algorithm_config.num_runs_per_config,
            evolutionary_algorithm_config=evolutionary_algorithm_config,
            engine_config1=engine_config1,
            engine_config2=engine_config2,
            command_line_args=args,
            logger=logger,
        )
    )

    # Average the fitness values and unique individual fractions over all runs
    # and compute the standard deviation
    average_runtime = np.mean(run_times)
    best_fitness_values = np.mean(best_fitness_value_series, axis=0)
    average_fitness_values = np.mean(average_fitness_value_series, axis=0)
    worst_fitness_values = np.mean(worst_fitness_value_series, axis=0)
    unique_individual_fraction = np.mean(unique_individual_fractions, axis=0)
    best_fitness_values_std = np.std(best_fitness_value_series, axis=0)
    average_fitness_values_std = np.std(average_fitness_value_series, axis=0)
    worst_fitness_values_std = np.std(worst_fitness_value_series, axis=0)
    unique_individual_fraction_std = np.std(unique_individual_fractions, axis=0)

    # Log the results
    if not DEBUG:
        for i in range(len(best_fitness_values)):
            wandb.log(
                {
                    "best_fitness_value_avg": best_fitness_values[i],
                    "best_fitness_value_std": best_fitness_values_std[i],
                    "average_fitness_value_avg": average_fitness_values[i],
                    "average_fitness_value_std": average_fitness_values_std[i],
                    "worst_fitness_value_avg": worst_fitness_values[i],
                    "worst_fitness_value_std": worst_fitness_values_std[i],
                    "unique_individual_fraction_avg": unique_individual_fraction[i],
                    "unique_individual_fraction_std": unique_individual_fraction_std[i],
                },
                step=i,
            )

        wandb.log({"average_runtime": average_runtime})
