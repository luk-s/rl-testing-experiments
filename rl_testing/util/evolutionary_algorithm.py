from typing import Any, Callable, List, Optional, Tuple

from rl_testing.evolutionary_algorithms.individuals import Individual


def clear_fitness_values_wrapper(
    function: Callable[[Individual, Any], Individual]
) -> Callable[[Individual, Any], Individual]:
    """Wrapper for mutation functions that clears the fitness values of the mutated individual.

    Args:
        function (Callable[[Individual, Any], Individual]): The mutation function to wrap.

    Returns:
        Callable[[Individual, Any], Individual]: The wrapped mutation function.
    """

    def inner_function(individual: Individual, *args: Any, **kwargs: Any) -> Individual:
        # Call the mutation function
        mutated_individual = function(individual, *args, **kwargs)

        # Clear the fitness values
        del mutated_individual.fitness

        return mutated_individual

    inner_function.__name__ = function.__name__
    return inner_function
