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


def __should_decrease_probability(
    fitness: float,
    difference_threshold: float,
    __fitness_history: List[float] = [],
) -> bool:
    """Check if the probability of a mutation or crossover should be decreased. This function should not be called
    by the user.

    ATTENTION! This makes use of the peculiar behavior of Python that default arguments are bound
    at the time of the function definition and not at the time of the function call. See
    https://stackoverflow.com/questions/9158294/good-uses-for-mutable-function-argument-default-values
    This means that the default argument is shared between all calls of the function.
    In this case this is exactly what we want because we want to keep track of the fitness history

    Args:
        best_fitness (float): The best fitness value of the current generation.
        difference_threshold (float): The threshold for the difference between the best and worst fitness value of the last
            10 generations. If the difference is smaller than this threshold, the probability should be decreased.
        __fitness_history (List[float]): The fitness history of the current generation. Should not be provided by the user.

    Returns:
        bool: True if the probability should be decreased, False otherwise.
    """
    # Add the best fitness value of the current generation to the history
    __fitness_history.append(fitness)

    if len(__fitness_history) < 10:
        return False

    # Get the maximum and minimum fitness value of the last 10 generations
    largest_fitness = max(__fitness_history[-10:])
    smallest_fitness = min(__fitness_history[-10:])

    # Check if the difference between the best and worst fitness value of the last 10 generations is smaller than the
    # threshold
    if abs(largest_fitness - smallest_fitness) <= difference_threshold:
        __fitness_history.clear()
        return True

    return False


def should_decrease_probability(best_fitness: float, difference_threshold: float):
    """Check if the probability of a mutation or crossover should be decreased.

    Args:
        best_fitness (float): The best fitness value of the current generation.
        difference_threshold (float): The threshold for the difference between the best and worst fitness value of the last
            10 generations. If the difference is smaller than this threshold, the probability should be decreased.

    Returns:
        bool: True if the probability should be decreased, False otherwise.
    """
    return __should_decrease_probability(best_fitness, difference_threshold)
