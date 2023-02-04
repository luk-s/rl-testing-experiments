import logging
import time
from operator import attrgetter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from rl_testing.evolutionary_algorithms.individuals import Individual
from rl_testing.util.util import get_random_state


def select_tournament_fast(
    individuals: List[Individual],
    rounds: int,
    tournament_size: int,
    is_bigger_better: bool = True,
    batch_size: int = 1000,
    _random_state: Optional[np.random.Generator] = None,
) -> List[Individual]:
    """
    This is an optimized version of the simple implementation found in the deap library:
    https://github.com/DEAP/deap/blob/master/deap/tools/selection.py

    Select the best individual among "tournament_size" randomly chosen
    individuals, "rounds" times. The list returned contains
    references to the input "individuals".

    Args:
        individuals (List[Individual]): A list of individuals to select from.
        rounds (int): The number of times to select an individual.
        tournament_size (int): The number of individuals to choose from.
        is_bigger_better (bool, optional): Whether a higher fitness value is better. Defaults to True.
        batch_size (int, optional): The number of individuals to select at a time. Defaults to 1000.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[Individual]: The selected individuals.
    """
    random_state = get_random_state(_random_state)

    # If the batch size is larger than 1000, warn the user
    if batch_size > 1000:
        logging.warning(
            f"Batch size is set to {batch_size}, which is larger than 1000. This may take a long time and consume a lot of memory."
        )

    # First sort the individuals by fitness.
    individuals.sort(key=attrgetter("fitness"), reverse=is_bigger_better)

    # Compute the individual batch sizes
    batch_sizes = [batch_size] * (rounds // batch_size) + [rounds % batch_size]

    # Select the individuals
    # This does the same like the function 'select_tournament' but is much faster.
    tournament_winner_indices = []
    for current_batch_size in batch_sizes:
        # Compute a 2d array of random numbers
        random_numbers = random_state.random((current_batch_size, len(individuals)))

        # For each row, find the indices of the 'tournament_size' largest random numbers
        tournament_indices = np.argpartition(random_numbers, tournament_size, axis=1)[
            :, -tournament_size:
        ]

        # For each row, find the smallest number and append it to the list of tournament winners
        tournament_winner_indices.extend(np.min(tournament_indices, axis=1))

    # Return the tournament winners
    return [individuals[i] for i in tournament_winner_indices]


def select_tournament(
    individuals: List[Individual],
    rounds: int,
    tournament_size: int,
    find_best_individual: Callable[[List[Individual]], Tuple[Individual, float]],
    _random_state: Optional[np.random.Generator] = None,
) -> List[Individual]:
    """
    Adapted from the deap library: https://github.com/DEAP/deap/blob/master/deap/tools/selection.py
    Select the best individual among "tournament_size" randomly chosen
    individuals, "rounds" times. The list returned contains
    references to the input "individuals".

    Args:
        individuals (List[Individual]): A list of individuals to select from.
        rounds (int): The number of times to select an individual.
        tournament_size (int): The number of individuals to choose from.
        find_best_individual (Callable[[List[Individual]], Tuple[Individual, float]]): A function that finds the best
            individual in a list of individuals.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[Individual]: The selected individuals.

    """
    random_state = get_random_state(_random_state)

    chosen = []
    start_time = time.time()
    for i in range(rounds):
        aspirants = random_state.choice(individuals, tournament_size, replace=False)
        chosen.append(find_best_individual(aspirants)[0])
    return chosen


class Selector:
    def __init__(
        self,
        selection_strategy: str = "all",
        num_selection_functions: Optional[int] = None,
        _random_state: Optional[np.random.Generator] = None,
    ):
        """A class that selects individuals from a population.

        Args:
            selection_strategy (str, optional): The selection strategy to use. Must be one of
                ["all", "one_random", "n_random"] where "all" applies all selection functions, "one_random" applies one
                random selection function, and "n_random" applies "num_selection_functions" random selection functions.
                Defaults to "all".
            num_selection_functions (Optional[int], optional): The number of selection functions to apply if the selection
                strategy is "n_random". Defaults to None.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
        """
        self.selection_function_tuples: List[
            Tuple[Callable[[Individual, Any], List[Individual]], List[Any], Dict[str, Any]]
        ] = []
        self.selection_strategy = selection_strategy
        self.num_selection_functions = num_selection_functions
        self.random_state = get_random_state(_random_state)

        assert self.selection_strategy in [
            "all",
            "one_random",
            "n_random",
        ], f"Selection strategy must be one of ['all', 'one_random', 'n_random'], got {self.selection_strategy}."

        if self.selection_strategy == "n_random":
            assert (
                self.num_selection_functions is not None
            ), "Number of selections must be specified if selection strategy is 'n_random'."

    def register_selection_function(
        self,
        selection_function: Callable[[Individual, Any], List[Individual]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a selection function.

        Args:
            selection_function (function): The selection function to register.
        """
        self.selection_function_tuples.append((selection_function, args, kwargs))

    def __call__(
        self, individuals: List[Individual], *new_args: Any, **new_kwargs: Any
    ) -> List[Individual]:
        """Select individuals from a population according to the selection strategy.

        Args:
            individuals (List[Individual]): The population to select from.

        Raises:
            ValueError: If the selection strategy is invalid.

        Returns:
            List[Individual]: The selected individuals.
        """
        mutation_tuples_to_apply: List[
            Tuple[Callable[[Individual, Any], List[Individual]], List[Any], Dict[str, Any]]
        ]

        if self.selection_strategy == "all":
            mutation_tuples_to_apply = self.selection_function_tuples
        elif self.selection_strategy == "one_random":
            mutation_tuples_to_apply = [self.random_state.choice(self.selection_function_tuples)]
        elif self.selection_strategy == "n_random":
            mutation_tuples_to_apply = self.random_state.choice(
                self.selection_function_tuples, self.num_selection_functions, replace=False
            )
        else:
            raise ValueError(f"Invalid selection strategy {self.selection_strategy}.")

        selected_individuals = []
        for selection_function, args, kwargs in mutation_tuples_to_apply:
            selected_individuals.extend(
                selection_function(
                    individuals,
                    _random_state=self.random_state,
                    *args,
                    *new_args,
                    **kwargs,
                    **new_kwargs,
                )
            )

        return selected_individuals


if __name__ == "__main__":
    import time

    from rl_testing.evolutionary_algorithms.fitness import EditDistanceFitness
    from rl_testing.evolutionary_algorithms.individuals import BoardIndividual

    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    population_size = 14000
    tournament_fraction = 0.6969565124339143

    # Create a list of fake individuals
    individuals = [BoardIndividual() for _ in range(population_size)]

    # Assign random fitness values to the individuals
    for individual in individuals:
        individual.fitness = np.random.random()

    # Create two selectors
    selector_simple = Selector(selection_strategy="all")
    selector_fast = Selector(selection_strategy="all")

    # Register the selection functions
    fitness = EditDistanceFitness("3r3k/7p/2p1np2/4p1p1/1Pq1P3/2Q2P2/P4RNP/2R4K b - - 0 42")
    selector_simple.register_selection_function(
        select_tournament,
        rounds=population_size,
        tournament_size=int(population_size * tournament_fraction),
        find_best_individual=fitness.best_individual,
    )
    selector_fast.register_selection_function(
        select_tournament_fast,
        rounds=population_size,
        tournament_size=int(population_size * tournament_fraction),
        is_bigger_better=False,
    )

    # Select individuals
    start_time = time.time()
    selected_individuals_simple = selector_simple(individuals)
    logging.info(f"Time simple selection: {round(time.time() - start_time,3)} seconds.")
    start_time = time.time()
    selected_individuals_fast = selector_fast(individuals)
    logging.info(f"Time fast selection: {round(time.time() - start_time,3)} seconds.")

    # Print the average fitness of the selected individuals
    logging.info(
        f"Average fitness simple: {np.mean([individual.fitness for individual in selected_individuals_simple])}"
    )
    logging.info(
        f"Average fitness fast: {np.mean([individual.fitness for individual in selected_individuals_fast])}"
    )
