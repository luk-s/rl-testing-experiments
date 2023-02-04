import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chess
import numpy as np

from rl_testing.evolutionary_algorithms.individuals import BoardIndividual, Individual
from rl_testing.util.chess import is_really_valid
from rl_testing.util.util import get_random_state


def _ensure_single_kings(
    board: chess.Board,
    random_state: np.random.Generator,
) -> chess.Board:
    """Ensure that the board has only one king per color.

    Args:
        board (chess.Board): The board.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The board with only one king per color.
    """

    for color in [chess.WHITE, chess.BLACK]:
        # Get the king squares
        squares = [
            square
            for square, piece in board.piece_map().items()
            if piece.piece_type == chess.KING and piece.color == color
        ]

        # If there are more than two kings per color, randomly remove one of them
        if len(squares) > 1:
            board.remove_piece_at(random_state.choice(squares))

        # If there are no kings, randomly place one
        while len(squares) == 0:
            piece_map = board.piece_map()

            # Find all empty squares which are not attacked by the opposing color
            empty_squares = [
                square
                for square in chess.SQUARES
                if board.piece_at(square) is None and not board.is_attacked_by(not color, square)
            ]

            # If there are no empty squares, randomly remove a piece
            if len(empty_squares) == 0:
                board.remove_piece_at(random_state.choice(list(piece_map.keys())))
            else:
                board.set_piece_at(
                    random_state.choice(empty_squares),
                    chess.Piece(chess.KING, color),
                )
                break

    # Assert that there is exactly one king per color
    pieces = board.piece_map().values()
    for color in [chess.WHITE, chess.BLACK]:
        color_name = "white" if color == chess.WHITE else "black"
        num_kings = sum(
            [piece.piece_type == chess.KING and piece.color == color for piece in pieces]
        )
        assert num_kings == 1, f"Board has {num_kings} {color_name} kings."

    return board


def crossover_half_board(
    board1: chess.Board,
    board2: chess.Board,
    axis: Optional[str] = None,
    ensure_single_kings: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Tuple[chess.Board, chess.Board]:
    """Crossover function that swaps the horizontal halves of the two boards.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
        axis (Optional[str], optional): Whether to swap one of the horizontal halves or the vertical halves.
            Must be either "horizontal" or "vertical". Defaults to None in which case it is chosen randomly.
        ensure_single_kings (bool, optional): Whether to ensure that the boards have only one king per color after the crossover.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Tuple[chess.Board, chess.Board]: The two boards after the crossover.
    """

    random_state = get_random_state(_random_state)

    if axis is None:
        axis = random_state.choice(["horizontal", "vertical"])

    assert axis in [
        "horizontal",
        "vertical",
    ], f"Axis must be either 'horizontal' or 'vertical', got {axis}."

    if axis == "horizontal":
        files = range(8)
        ranks = range(4)
    elif axis == "vertical":
        files = range(4)
        ranks = range(8)

    for rank in ranks:
        for file in files:
            piece1 = board1.piece_at(chess.square(file, rank))
            piece2 = board2.piece_at(chess.square(file, rank))
            board1.set_piece_at(chess.square(file, rank), piece2)
            board2.set_piece_at(chess.square(file, rank), piece1)

    if ensure_single_kings:
        board1 = _ensure_single_kings(board1, random_state)
        board2 = _ensure_single_kings(board2, random_state)

    logging.debug(f"Swapped {axis} halves of the boards.")

    return board1, board2


def crossover_one_quarter_board(
    board1: chess.Board,
    board2: chess.Board,
    ensure_single_kings: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Tuple[chess.Board, chess.Board]:
    """Crossover function that swaps the quarter boards of the two boards.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
        ensure_single_kings (bool, optional): Whether to ensure that the boards have only one king per color after the crossover.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Tuple[chess.Board, chess.Board]: The two boards after the crossover.
    """

    random_state = get_random_state(_random_state)

    # Randomly select one of the four quarters
    quarter = random_state.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    if quarter == "top_left":
        files = range(4)
        ranks = range(4, 8)
    elif quarter == "top_right":
        files = range(4, 8)
        ranks = range(4, 8)
    elif quarter == "bottom_left":
        files = range(4)
        ranks = range(4)
    elif quarter == "bottom_right":
        files = range(4, 8)
        ranks = range(4)

    for rank in ranks:
        for file in files:
            piece1 = board1.piece_at(chess.square(file, rank))
            piece2 = board2.piece_at(chess.square(file, rank))
            board1.set_piece_at(chess.square(file, rank), piece2)
            board2.set_piece_at(chess.square(file, rank), piece1)

    if ensure_single_kings:
        board1 = _ensure_single_kings(board1, random_state)
        board2 = _ensure_single_kings(board2, random_state)

    logging.debug(f"Swapped {quarter} quarter of the boards.")

    return board1, board2


def crossover_one_eighth_board(
    board1: chess.Board,
    board2: chess.Board,
    ensure_single_kings: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Tuple[chess.Board, chess.Board]:

    """Crossover function that swaps the eighth boards of the two boards.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
        ensure_single_kings (bool, optional): Whether to ensure that the boards have only one king per color after the crossover.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Tuple[chess.Board, chess.Board]: The two boards after the crossover.
    """

    random_state = get_random_state(_random_state)

    # Randomly select one of the eight quarters
    start_rank = random_state.choice(range(7))
    start_file = random_state.choice(range(7))

    files = range(start_file, start_file + 2)
    ranks = range(start_rank, start_rank + 2)

    for rank in ranks:
        for file in files:
            piece1 = board1.piece_at(chess.square(file, rank))
            piece2 = board2.piece_at(chess.square(file, rank))
            board1.set_piece_at(chess.square(file, rank), piece2)
            board2.set_piece_at(chess.square(file, rank), piece1)

    if ensure_single_kings:
        board1 = _ensure_single_kings(board1, random_state)
        board2 = _ensure_single_kings(board2, random_state)

    logging.debug(f"Swapped ({start_rank},{start_file}) eighth of the boards.")

    return board1, board2


def validity_wrapper(
    function: Callable[[chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]],
    retries: int = 0,
) -> Callable[[chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]]:
    """Wrapper for the crossover functions that checks if the input boards are valid after the crossover.
    If the boards aren't valid, the original boards are returned.

    Args:
        function (function): The crossover function to wrap.
        retries (int, optional): The number of times to retry the crossover if the board is invalid. Defaults to 0.
        *args: The arguments to pass to the crossover function.
        **kwargs: The keyword arguments to pass to the crossover function.

    Returns:
        inner_function (function): The wrapped crossover function.
    """

    def inner_function(
        board1: chess.Board, board2: chess.Board, *args: Any, **kwargs: Any
    ) -> chess.Board:
        for _ in range(retries + 1):
            # Clone the original board
            board_candidate1 = board1.copy()
            board_candidate2 = board2.copy()

            # Retry the crossover if the board is invalid
            board_candidate1, board_candidate2 = function(
                board_candidate1, board_candidate2, *args, **kwargs
            )

            # Check if the board is valid
            if is_really_valid(board_candidate1) and is_really_valid(board_candidate2):
                return board_candidate1, board_candidate2

        logging.debug(
            f"Board {board_candidate1.fen()} or Board {board_candidate2.fen()}"
            f" is invalid after crossover '{function.__name__}', returning original boards"
        )
        return board1, board2

    return inner_function


def clear_fitness_values_wrapper(
    function: Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]
) -> Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]:
    """Wrapper for crossover functions that clears the fitness values of the mated individuals.

    Args:
        function (Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]): The crossover function to wrap.
    Returns:
        Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]: The wrapped crossover function.
    """

    def inner_function(
        individual1: Individual, individual2: Individual, *args: Any, **kwargs: Any
    ) -> Tuple[Individual, Individual]:
        # Call the crossover function
        crossed_individual1, crossed_individual2 = function(
            individual1, individual2, *args, **kwargs
        )

        # Clear the fitness values
        del crossed_individual1.fitness
        del crossed_individual2.fitness

        return crossed_individual1, crossed_individual2

    return inner_function


def print_side_by_side(board1: chess.Board, board2: chess.Board) -> None:
    """Prints the two boards side by side.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
    """
    board1_ranks = str(board1).split("\n")
    board2_ranks = str(board2).split("\n")

    for rank1, rank2 in zip(board1_ranks, board2_ranks):
        print(rank1, "\t", rank2)
    print("\n")


class CrossoverFunction:
    def __init__(
        self,
        function: Callable[[chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]],
        probability: float = 1.0,
        retries: int = 0,
        check_game_not_over: bool = False,
        clear_fitness_values: bool = False,
        _random_state: Optional[np.random.Generator] = None,
        *args,
        **kwargs,
    ):
        """A convenience class for storing crossover functions together with some settings.
        Args:
            function (Callable[[chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]]): The crossover function.
            retries (int, optional): The number of times to retry the crossover if the board is invalid. Defaults to 0.
            clear_fitness_values (bool, optional): Whether to clear the fitness values of the mated individual.
                Defaults to False.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None
            *args: Default arguments to pass to the crossover function.
            **kwargs: Default keyword arguments to pass to the crossover function.
        """
        self.probability = probability
        self.retries = retries
        self.random_state = get_random_state(_random_state)

        self.clear_fitness_values = clear_fitness_values
        self.check_game_not_over = check_game_not_over
        self.function = function

        self.args = args
        self.kwargs = kwargs

    def __call__(
        self, board1: BoardIndividual, board2: BoardIndividual, *new_args: Any, **new_kwargs: Any
    ) -> Tuple[BoardIndividual, BoardIndividual]:
        """Call the crossover function.

        Args:
            board1 (BoardIndividual): The first board to mate.
            board2 (BoardIndividual): The second board to mate.
            *new_args: Additional arguments to pass to the crossover function.
            **new_kwargs: Additional keyword arguments to pass to the crossover function.

        Returns:
            Tuple[BoardIndividual, BoardIndividual]: The mated boards.
        """
        for _ in range(self.retries + 1):
            # Clone the original board
            board_candidate1 = board1.copy()
            board_candidate2 = board2.copy()

            # Retry the crossover if the board is invalid
            board_candidate1, board_candidate2 = self.function(
                board_candidate1,
                board_candidate2,
                _random_state=self.random_state,
                *self.args,
                *new_args,
                **self.kwargs,
                **new_kwargs,
            )

            # Check if the board is valid
            if is_really_valid(board_candidate1) and is_really_valid(board_candidate2):
                if not self.check_game_not_over or (
                    len(list(board_candidate1.legal_moves)) > 0
                    and len(list(board_candidate2.legal_moves)) > 0
                ):
                    if self.clear_fitness_values:
                        del board_candidate1.fitness
                        del board_candidate2.fitness
                    return board_candidate1, board_candidate2

        logging.debug(
            f"Board {board_candidate1.fen()} or Board {board_candidate2.fen()}"
            f" is invalid after crossover '{self.function.__name__}', returning original boards"
        )
        return board1, board2


class Crossover:
    def __init__(
        self,
        crossover_strategy: str = "all",
        num_crossover_functions: Optional[int] = None,
        _random_state: Optional[np.random.Generator] = None,
    ):
        """Crossover class that can be used to perform crossover on a population.

        Args:
            crossover_strategy (str, optional): The strategy to use for the crossover. Must be one of
                ["all", "one_random", "n_random"] where "all" applies all crossover functions, "one_random" applies one
                random crossover function, and "n_random" applies "num_crossover_functions" random selection functions.
                Defaults to "all".
            num_crossover_functions (Optional[int], optional): The number of crossover functions to use if
                "crossover_strategy" is "n_random". Defaults to None.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
        """
        self.crossover_strategy = crossover_strategy
        self.num_crossover_functions = num_crossover_functions
        self.random_state = get_random_state(_random_state)
        self.global_probability: Optional[float] = None

        self.crossover_functions: List[CrossoverFunction] = []
        self.crossover_functions_dict: Dict[str, CrossoverFunction] = {}

        assert self.crossover_strategy in [
            "all",
            "one_random",
            "n_random",
        ], f"Invalid crossover strategy. Must be one of ['all', 'one_random', 'n_random'] but got {self.crossover_strategy}."
        if self.crossover_strategy == "n_random":
            assert (
                self.num_crossover_functions is not None
            ), "Must specify the number of crossover functions to use if crossover strategy is 'n_random'."

    def register_crossover_function(
        self,
        functions: Union[
            Callable[[Individual, Individual, Any], Tuple[Individual, Individual]],
            List[Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]],
        ],
        probability: float = 1.0,
        retries: int = 0,
        check_game_not_over: bool = False,
        clear_fitness_values: bool = True,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        """Registers a crossover function.

        Args:
            crossover_function (Union[
                Callable[[Individual, Individual, Any], Tuple[Individual, Individual]],
                List[Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]]]):
                The crossover function to register. Can be a single function or a list of functions.
            clear_fitness (bool, optional): Whether to clear the fitness of the individuals after the crossover. Defaults to True.
            args (List[Any], optional): The arguments to pass to the crossover function. Defaults to [].
            kwargs (Dict[str, Any], optional): The keyword arguments to pass to the crossover function. Defaults to {}.
        """
        if not isinstance(functions, list):
            functions = [functions]

        for function in functions:
            self.crossover_functions.append(
                CrossoverFunction(
                    function,
                    probability=probability,
                    retries=retries,
                    check_game_not_over=check_game_not_over,
                    clear_fitness_values=clear_fitness_values,
                    _random_state=self.random_state,
                    *args,
                    **kwargs,
                )
            )
            self.crossover_functions_dict[function.__name__] = self.crossover_functions[-1]

    def change_mutation_function_parameters(
        self,
        function_names: Union[str, List[str]],
        **kwargs: Any,
    ) -> None:
        """Changes the parameters of one or multiple mutation functions

        Args:
            function_names (str): The name of the mutation functions to change the parameters of.
            **kwargs: The new parameters to set.
        """
        if not isinstance(function_names, list):
            function_names = [function_names]

        # Separate the named arguments from the keyword arguments
        named_arg_tuples = []
        if "probability" in kwargs:
            named_arg_tuples.append(("probability", kwargs["probability"]))
            del kwargs["probability"]
        if "retries" in kwargs:
            named_arg_tuples.append(("retries", kwargs["retries"]))
            del kwargs["retries"]
        if "clear_fitness_values" in kwargs:
            named_arg_tuples.append(("clear_fitness_values", kwargs["clear_fitness_values"]))
            del kwargs["clear_fitness_values"]

        for function_name in function_names:
            function = self.crossover_functions_dict[function_name]

            # Set the named arguments
            for named_arg_tuple in named_arg_tuples:
                setattr(function, named_arg_tuple[0], named_arg_tuple[1])

            # Set the keyword arguments
            function.kwargs = {**function.kwargs, **kwargs}

    def set_global_probability(self, probability: float) -> None:
        """Sets the probability of all crossover functions.

        Args:
            probability (float): The probability to set.
        """
        self.global_probability = probability
        for crossover_function in self.crossover_functions:
            crossover_function.probability = probability

    def __call__(
        self, individual_tuple: Tuple[Individual, Individual], *new_args: Any, **new_kwargs: Any
    ) -> Tuple[Individual, Individual]:
        """Calls the crossover function on the two individuals according to the crossover strategy.

        Args:
            individual_tuple (Tuple[Individual, Individual]): The two individuals to mate.

        Returns:
            Tuple[Individual, Individual]: The two individuals after the crossover.
        """
        individual1, individual2 = individual_tuple
        crossover_functions_to_apply = []

        if self.crossover_strategy == "all":
            crossover_functions_to_apply = self.crossover_functions
        elif self.crossover_strategy == "one_random":
            crossover_functions_to_apply = [self.random_state.choice(self.crossover_functions)]
        elif self.crossover_strategy == "n_random":
            crossover_functions_to_apply = self.random_state.choice(
                self.crossover_functions, self.num_crossover_functions, replace=False
            )
        else:
            raise ValueError(
                f"Crossover strategy must be one of ['all', 'one_random', 'n_random'] but got {self.crossover_strategy}."
            )

        for crossover_function in crossover_functions_to_apply:
            if self.random_state.random() < crossover_function.probability:
                individual1, individual2 = crossover_function(
                    individual1,
                    individual2,
                    *new_args,
                    **new_kwargs,
                )

        return individual1, individual2


if __name__ == "__main__":
    # Configure logging to show debug messages
    logging.basicConfig(level=logging.DEBUG)

    board1 = chess.Board("8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118")
    board2 = chess.Board("r3qb1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 w - - 4 11")
    print_side_by_side(board1, board2)
    crossover_one_eighth_board(board1, board2, ensure_single_kings=True)
    print_side_by_side(board1, board2)
