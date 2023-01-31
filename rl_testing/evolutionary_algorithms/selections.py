from operator import attrgetter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from rl_testing.evolutionary_algorithms.individuals import Individual
from rl_testing.util.util import get_random_state


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
