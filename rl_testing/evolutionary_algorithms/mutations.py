import abc
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chess
import numpy as np
from rl_testing.evolutionary_algorithms.individuals import BoardIndividual, Individual
from rl_testing.util.chess import (
    is_really_valid,
    rotate_90_clockwise,
    rotate_180_clockwise,
    rotate_270_clockwise,
)
from rl_testing.util.evolutionary_algorithm import clear_fitness_values_wrapper
from rl_testing.util.util import get_random_state

PIECE_COUNT_DICT = {
    chess.WHITE: {"R": 2, "N": 2, "B": 2, "Q": 1, "K": 1, "P": 8},
    chess.BLACK: {"r": 2, "n": 2, "b": 2, "q": 1, "k": 1, "p": 8},
}


class MutationName(Enum):
    MUTATE_ADD_ONE_PIECE = 0
    MUTATE_CASTLING_RIGHTS = 1
    MUTATE_FLIP_BOARD = 2
    MUTATE_MOVE_ONE_PIECE = 3
    MUTATE_MOVE_ONE_PIECE_ADJACENT = 4
    MUTATE_MOVE_ONE_PIECE_LEGAL = 5
    MUTATE_PLAYER_TO_MOVE = 6
    MUTATE_REMOVE_ONE_PIECE = 7
    MUTATE_ROTATE_BOARD = 8
    MUTATE_SUBSTITUTE_PIECE = 9


def mutate_player_to_move(
    board: chess.Board, _random_state: Optional[np.random.Generator] = None
) -> chess.Board:
    """Flip the player to move.

    Args:
        board (chess.Board): The board to mutate.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """

    random_state = get_random_state(_random_state)

    board.push(chess.Move.null())

    return board


def mutate_castling_rights(
    board: chess.Board,
    probability_per_direction: float = 0.5,
    _random_state: Optional[np.random.Generator] = None,
) -> chess.Board:
    """Mutate each castling right with probability `probability`.

    Args:
        board (chess.Board): The board to mutate.
        probability (float): The probability of mutating the board. Defaults to 0.5.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """
    assert (
        0 <= probability_per_direction <= 1
    ), f"Probability must be between 0 and 1, got {probability_per_direction}"

    random_state = get_random_state(_random_state)

    # First check which of the four possible castling rights are theoretically possible
    possible_castling_rights = ""

    # Check castling possibilities for white
    if board.king(chess.WHITE) == chess.E1:
        if board.rooks & chess.BB_H1:
            possible_castling_rights += "K"
        if board.rooks & chess.BB_A1:
            possible_castling_rights += "Q"

    # Check castling possibilities for black
    if board.king(chess.BLACK) == chess.E8:
        if board.rooks & chess.BB_H8:
            possible_castling_rights += "k"
        if board.rooks & chess.BB_A8:
            possible_castling_rights += "q"

    # Now mutate the castling rights
    new_castling_rights = ""
    if possible_castling_rights:
        # Get the castling fen of the input board
        castling_fen = board.fen().split(" ")[2]

        # Mutate the castling rights
        for castling_right in possible_castling_rights:
            if random_state.random() < probability_per_direction:
                if castling_right not in castling_fen:
                    new_castling_rights += castling_right
            else:
                if castling_right in castling_fen:
                    new_castling_rights += castling_right

    # Set the new castling rights
    board.set_castling_fen(new_castling_rights)

    return board


def mutate_add_one_piece(
    board: chess.Board,
    color: Optional[chess.Color] = None,
    max_tries: int = 10,
    _random_state: Optional[np.random.Generator] = None,
) -> chess.Board:
    """Add one missing piece. If there are no missing pieces, do nothing.

    Args:
        board (chess.Board): The board to mutate.
        color (Optional[chess.Color], optional): The color of the piece to add. Defaults to None which means choose randomly.
        max_tries (int, optional): The maximum number of tries to find a valid square. Defaults to 10.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """

    random_state = get_random_state(_random_state)

    # Assign a color if none is given
    if color is None:
        color = random_state.choice([chess.WHITE, chess.BLACK])

    # Get the missing pieces for the given color
    pieces_dict = board.piece_map()
    piece_count = dict(PIECE_COUNT_DICT[color])
    for square in pieces_dict:
        if pieces_dict[square].color == color:
            piece_count[pieces_dict[square].symbol()] -= 1
    missing_pieces = [piece for piece, count in piece_count.items() if count > 0]

    empty_squares = list(set(chess.SQUARES) - set(pieces_dict.keys()))

    # Add one missing piece for the selected color at random
    if missing_pieces and sum(piece_count.values()) > 0:
        # It's not enough to just add a piece at a random empty square, because of additional
        # rules like e.g. a pawn can't be added to the first or last rank. So we repeat it until
        # we find a valid square.
        for _ in range(max_tries):
            piece = random_state.choice(missing_pieces)
            square = random_state.choice(empty_squares)
            board.set_piece_at(square, chess.Piece.from_symbol(piece))
            if is_really_valid(board):
                logging.debug(f"Added {piece} to {chess.square_name(square)}\n")
                break
            board.remove_piece_at(square)

    return board


def mutate_remove_one_piece(
    board: chess.Board,
    color: Optional[chess.Color] = None,
    _random_state: Optional[np.random.Generator] = None,
) -> chess.Board:
    """Remove one piece. If there are no pieces, do nothing.

    Args:
        board (chess.Board): The board to mutate.
        color (Optional[chess.Color], optional): The color of the piece to remove. Defaults to None which means choose randomly.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """

    random_state = get_random_state(_random_state)

    color_chosen = color is not None

    # Assign a color if none is given
    if not color_chosen:
        color = random_state.choice([chess.WHITE, chess.BLACK])

    # Get the squares with pieces for the given color
    pieces_dict = board.piece_map()
    squares = [square for square in pieces_dict if pieces_dict[square].color == color]

    # If squares is empty, and the color has not been specified directly by the user, try using the other color
    if not squares and not color_chosen:
        color = not color
        squares = [square for square in pieces_dict if pieces_dict[square].color == color]

    # Remove one piece for the selected color at random
    if squares:
        square = random_state.choice(squares)
        board.remove_piece_at(square)
        logging.debug(f"Removed piece from {chess.square_name(square)}\n")

    return board


def mutate_move_one_piece(
    board: chess.Board,
    color: Optional[chess.Color] = None,
    max_tries: int = 10,
    _random_state: Optional[np.random.Generator] = None,
) -> chess.Board:
    """Move one piece to an empty square. This doesn't need to be a legal move.

    Args:
        board (chess.Board): The board to mutate.
        color (Optional[chess.Color], optional): The color of the piece to move. Defaults to None which means choose randomly.
        max_tries (int, optional): The maximum number of tries to find a valid square. Defaults to 10.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """

    random_state = get_random_state(_random_state)

    # Assign a color if none is given
    if color is None:
        color = random_state.choice([chess.WHITE, chess.BLACK])

    # Get the pieces for the given color
    pieces_dict = board.piece_map()

    # Filter all pieces of the given color
    pieces_dict = {square: piece for square, piece in pieces_dict.items() if piece.color == color}

    # Filter all pieces that are not pinned
    pieces_dict = {
        square: piece
        for square, piece in pieces_dict.items()
        if not board.is_pinned(piece.color, square)
    }

    # It's not enough to just move a piece to a random adjacent square, because this might lead to
    # a check which might be illegal if the same color is to move. So we repeat it until we find a
    # valid move.
    if pieces_dict:
        for _ in range(max_tries):
            start_square = random_state.choice(list(pieces_dict.keys()))
            piece = pieces_dict[start_square]

            # Get all empty squares
            empty_squares = list(set(chess.SQUARES) - set(board.piece_map().keys()))

            # Move the selected piece to a random square
            if empty_squares:
                piece = board.remove_piece_at(start_square)
                target_square = random_state.choice(empty_squares)
                board.set_piece_at(target_square, piece)

                # Check validity of new position
                if is_really_valid(board):
                    logging.debug(
                        f"Moved {piece.symbol()} from {chess.square_name(start_square)} to {chess.square_name(target_square)}\n"
                    )
                    break
                else:
                    # Undo the move if it is not valid
                    board.remove_piece_at(target_square)
                    board.set_piece_at(start_square, piece)

    return board


def mutate_move_one_piece_legal(
    board: chess.Board,
    color: Optional[chess.Color] = None,
    _random_state: Optional[np.random.Generator] = None,
) -> chess.Board:
    """Move one piece by performing a legal move.

    Args:
        board (chess.Board): The board to mutate.
        color (Optional[chess.Color], optional): The color of the piece to move. Defaults to None which means choose randomly.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """
    random_state = get_random_state(_random_state)

    # Assign a color if none is given
    color_provided = color is not None
    if not color_provided:
        color = random_state.choice([chess.WHITE, chess.BLACK])

    # Temporarily set the board to the given color
    real_color = board.turn
    board.turn = color

    # Get all legal moves for the given color
    legal_moves_color = list(board.legal_moves)

    # If only one color has legal moves, and the user didn't select a color, switch to that color
    if not legal_moves_color:
        color = not color
        legal_moves_color = list(board.legal_moves)
        if not legal_moves_color or color_provided:
            board.turn = real_color
            return board

    # Move one piece at random
    move = random_state.choice(legal_moves_color)
    board.push(move)
    logging.debug(f"Moved {move}\n")

    board.turn = real_color
    return board


def mutate_move_one_piece_adjacent(
    board: chess.Board,
    color: Optional[chess.Color] = None,
    max_tries: int = 10,
    _random_state: Optional[np.random.Generator] = None,
) -> chess.Board:
    """Move one piece to an adjacent board square. This doesn't need to be a legal move.

    Args:
        board (chess.Board): The board to mutate.
        color (Optional[chess.Color], optional): The color of the piece to move. Defaults to None which means choose randomly.
        max_tries (int, optional): The maximum number of tries to find a valid square. Defaults to 10.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """

    random_state = get_random_state(_random_state)

    # Assign a color if none is given
    if color is None:
        color = random_state.choice([chess.WHITE, chess.BLACK])

    # Get the pieces for the given color
    pieces_dict = board.piece_map()

    # Filter all pieces of the given color
    pieces_dict = {square: piece for square, piece in pieces_dict.items() if piece.color == color}

    # Filter all pieces that are not pinned
    pieces_dict = {
        square: piece
        for square, piece in pieces_dict.items()
        if not board.is_pinned(piece.color, square)
    }

    # It's not enough to just move a piece to a random adjacent square, because this might lead to
    # a check which might be illegal if the same color is to move. So we repeat it until we find a
    # valid move.
    if pieces_dict:
        for _ in range(max_tries):
            start_square = random_state.choice(list(pieces_dict.keys()))
            piece = pieces_dict[start_square]

            # Get all empty squares
            empty_squares = list(set(chess.SQUARES) - set(board.piece_map().keys()))

            # Get all squares that are adjacent to the piece
            adjacent_squares = [
                s for s in empty_squares if chess.square_distance(start_square, s) == 1
            ]

            # Move the selected piece to a random adjacent square
            if adjacent_squares:
                piece = board.remove_piece_at(start_square)
                target_square = random_state.choice(adjacent_squares)
                board.set_piece_at(target_square, piece)

                # Check validity of new position
                if is_really_valid(board):
                    logging.debug(
                        f"Moved {piece.symbol()} from {chess.square_name(start_square)} to {chess.square_name(target_square)}\n"
                    )
                    break
                else:
                    # Undo the move if it is not valid
                    board.remove_piece_at(target_square)
                    board.set_piece_at(start_square, piece)

    return board


def mutate_rotate_board(
    board: chess.Board,
    angle: Optional[int] = None,
    _random_state: Optional[np.random.Generator] = None,
) -> chess.Board:
    """Rotate the board in the clockwise direction.

    Args:
        board (chess.Board): The board to mutate.
        angle (Optional[int], optional): The angle to rotate the board. Must be one of [90, 180, 270].
            Defaults to None which means choose randomly.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """

    random_state = get_random_state(_random_state)

    # Assign an angle if none is given
    if angle is None:
        angle = random_state.choice([90, 180, 270])

    assert angle in [90, 180, 270], f"Angle must be one of [90, 180, 270], got {angle}"

    # Rotate the board
    if angle == 90:
        board.apply_transform(rotate_90_clockwise)
    elif angle == 180:
        board.apply_transform(rotate_180_clockwise)
    elif angle == 270:
        board.apply_transform(rotate_270_clockwise)

    logging.debug(f"Rotated board by {angle} degrees\n")

    return board


def mutate_flip_board(
    board: chess.Board,
    axis: Optional[str] = None,
    _random_state: Optional[np.random.Generator] = None,
) -> chess.Board:
    """Flip the board along the given axis.

    Args:
        board (chess.Board): The board to mutate.
        axis (Optional[str], optional): The axis to flip the board. Must be one of
            ["horizontal", "vertical", "diagonal", "antidiagonal"].
            Defaults to None which means choose randomly.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The mutated board.
    """

    random_state = get_random_state(_random_state)

    # Assign an axis if none is given
    if axis is None:
        axis = random_state.choice(["horizontal", "vertical", "diagonal", "antidiagonal"])

    assert axis in [
        "horizontal",
        "vertical",
        "diagonal",
        "antidiagonal",
    ], f"Axis must be one of ['horizontal', 'vertical', 'diagonal', 'antidiagonal'], got {axis}"

    # Flip the board
    if axis == "horizontal":
        board.apply_transform(chess.flip_horizontal)
    elif axis == "vertical":
        board.apply_transform(chess.flip_vertical)
    elif axis == "diagonal":
        board.apply_transform(chess.flip_diagonal)
    elif axis == "antidiagonal":
        board.apply_transform(chess.flip_anti_diagonal)

    logging.debug(f"Flipped board along the {axis} axis\n")

    return board


def mutate_substitute_piece(
    board: chess.Board,
    _random_state: Optional[np.random.Generator] = None,
):
    """Substitute a piece on the board with another piece of arbitrary type and same color.

    Args:
        board (chess.Board): The board to mutate.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
    """
    random_state = get_random_state(_random_state)

    # Make sure that the board is valid. This allows us to assume that there are at least two kings on the board.
    assert is_really_valid(board), "Board is not valid"

    # Get all pieces on the board
    pieces_dict = board.piece_map()

    # Filter out kings
    pieces_dict = {k: v for k, v in pieces_dict.items() if v.symbol() not in ["k", "K"]}

    # If there are more than just kings on the board, select a random piece to substitute
    if pieces_dict:
        # Select a random piece
        square, piece = random_state.choice(list(pieces_dict.items()))

        # Select a random piece type
        piece_type = random_state.choice(list(chess.PIECE_TYPES[:-1]))

        # Create a new piece with the same color as the original piece
        new_piece = chess.Piece(piece_type, piece.color)

        # Remove the original piece
        board.remove_piece_at(square)

        # Add the new piece
        board.set_piece_at(square, new_piece)

        logging.debug(f"Substituted {piece.symbol()} with {new_piece.symbol()} at {square}\n")

    return board


def validity_wrapper(
    function: Callable[[chess.Board, Any], chess.Board],
    retries: int = 0,
) -> Callable[[chess.Board, Any], chess.Board]:
    """Wrapper for the mutate functions that checks if the board is valid after the mutation. If the board isn't
    valid, the original board is returned.

    Args:
        function (function): The mutate function to wrap.
        retries (int, optional): The number of times to retry the mutation if the board is invalid. Defaults to 0.
        *args: The arguments to pass to the mutate function.
        **kwargs: The keyword arguments to pass to the mutate function.

    Returns:
        inner_function (function): The wrapped mutate function.
    """

    def inner_function(board: chess.Board, *args: Any, **kwargs: Any) -> chess.Board:
        for _ in range(retries + 1):
            # Clone the original board
            board_candidate = board.copy()

            # Retry the mutation if the board is invalid
            board_candidate = function(board_candidate, *args, **kwargs)

            # Check if the board is valid
            if is_really_valid(board_candidate):
                return board_candidate

        logging.debug(
            f"Board {board_candidate.fen()} is invalid after mutation '{function.__name__}', returning original board"
        )
        return board

    inner_function.__name__ = function.__name__
    return inner_function


MUTATION_NAME_MAP = {
    mutate_add_one_piece: MutationName.MUTATE_ADD_ONE_PIECE,
    mutate_castling_rights: MutationName.MUTATE_CASTLING_RIGHTS,
    mutate_flip_board: MutationName.MUTATE_FLIP_BOARD,
    mutate_move_one_piece: MutationName.MUTATE_MOVE_ONE_PIECE,
    mutate_move_one_piece_adjacent: MutationName.MUTATE_MOVE_ONE_PIECE_ADJACENT,
    mutate_move_one_piece_legal: MutationName.MUTATE_MOVE_ONE_PIECE_LEGAL,
    mutate_player_to_move: MutationName.MUTATE_PLAYER_TO_MOVE,
    mutate_remove_one_piece: MutationName.MUTATE_REMOVE_ONE_PIECE,
    mutate_rotate_board: MutationName.MUTATE_ROTATE_BOARD,
    mutate_substitute_piece: MutationName.MUTATE_SUBSTITUTE_PIECE,
}


class MutationFunction:
    def __init__(
        self,
        function: Callable[[chess.Board, Any], chess.Board],
        probability: float = 1.0,
        retries: int = 0,
        check_game_not_over: bool = False,
        clear_fitness_values: bool = False,
        _random_state: Optional[np.random.Generator] = None,
        *args,
        **kwargs,
    ):
        """A convenience class for storing mutation functions together with some settings.
        Args:
            function (Callable[[chess.Board, Any], chess.Board]): The mutation function.
            retries (int, optional): The number of times to retry the mutation if the board is invalid. Defaults to 0.
            clear_fitness_values (bool, optional): Whether to clear the fitness values of the mutated individual.
                Defaults to False.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None
            *args: Default arguments to pass to the mutate function.
            **kwargs: Default keyword arguments to pass to the mutate function.
        """
        self.probability = probability
        self.retries = retries
        self.random_state = get_random_state(_random_state)

        self.function = function
        self.clear_fitness_values = clear_fitness_values
        self.check_game_not_over = check_game_not_over

        self.args = args
        self.kwargs = kwargs

    def __call__(
        self, board: BoardIndividual, *new_args: Any, **new_kwargs: Any
    ) -> BoardIndividual:
        """Call the mutation function.

        Args:
            board (BoardIndividual): The board to mutate.
            *new_args: Additional arguments to pass to the mutate function.
            **new_kwargs: Additional keyword arguments to pass to the mutate function.

        Returns:
            BoardIndividual: The mutated board.
        """
        for _ in range(self.retries + 1):
            # Clone the original board
            board_candidate = board.copy()

            # Retry the mutation if the board is invalid
            board_candidate = self.function(
                board_candidate,
                _random_state=self.random_state,
                *self.args,
                *new_args,
                **self.kwargs,
                **new_kwargs,
            )

            # Check if the board is valid
            if is_really_valid(board_candidate) and (
                not self.check_game_not_over or len(list(board_candidate.legal_moves)) > 0
            ):
                # Remove the fitness values if requested
                if self.clear_fitness_values:
                    del board_candidate.fitness

                # Add the applied mutation to the boards history
                board_candidate.history.append(MUTATION_NAME_MAP[self.function])

                return board_candidate

        logging.debug(
            f"Board {board_candidate.fen()} is invalid after mutation '{self.function.__name__}', returning original board"
        )
        return board


class MutationStrategy(metaclass=abc.ABCMeta):
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "__call__")) or NotImplemented

    def __init__(
        self,
        mutation_functions: List[MutationFunction],
        _random_state: Optional[np.random.Generator] = None,
    ):
        """A base class for mutation strategies.

        Args:
            mutation_functions (List[MutationFunction]): A **pointer** to the list of mutation functions.
            This means that the list of mutation functions can be modified externally and the changes
            will be reflected here.
        """
        self.random_state: np.random.Generator = get_random_state(_random_state)
        self.mutation_functions: List[MutationFunction] = mutation_functions

    @abc.abstractmethod
    def __call__(self, individual: BoardIndividual, *args: Any, **kwargs: Any) -> BoardIndividual:
        raise NotImplementedError


class AllMutationFunctionsStrategy(MutationStrategy):
    def __call__(self, individual: BoardIndividual, *args: Any, **kwargs: Any) -> BoardIndividual:
        """Apply all mutation functions to the individual.

        Args:
            individual (BoardIndividual): The individual to mutate.

        Returns:
            BoardIndividual: The mutated individual.
        """
        for mutation_function in self.mutation_functions:
            if self.random_state.random() < mutation_function.probability:
                individual = mutation_function(individual, *args, **kwargs)

        return individual


class OneRandomMutationFunctionStrategy(MutationStrategy):
    def __call__(self, individual: BoardIndividual, *args: Any, **kwargs: Any) -> BoardIndividual:
        """Apply one random mutation function to the individual. The probability of each mutation function
        is determined by the "probability" attribute of the mutation function.

        Args:
            individual (BoardIndividual): The individual to mutate.

        Returns:
            BoardIndividual: The mutated individual.
        """
        probabilities = [
            mutation_function.probability for mutation_function in self.mutation_functions
        ]
        mutation_function: MutationFunction = self.random_state.choice(
            self.mutation_functions, p=probabilities
        )
        individual = mutation_function(individual, *args, **kwargs)

        return individual


class NRandomMutationFunctionsStrategy(MutationStrategy):
    def __init__(
        self,
        mutation_functions: List[MutationFunction],
        num_mutation_functions: int,
        _random_state: Optional[np.random.Generator] = None,
    ):
        """A mutation strategy that applies "num_mutation_functions" random mutation functions to the individual.

        Args:
            mutation_functions (List[MutationFunction]): A **pointer** to the list of mutation functions.
                This means that the list of mutation functions can be modified externally and the changes
                will be reflected here.
            num_mutation_functions (int): The number of mutation functions to apply.
        """
        super().__init__(mutation_functions, _random_state)
        self.num_mutation_functions = num_mutation_functions

    def __call__(self, individual: BoardIndividual, *args: Any, **kwargs: Any) -> BoardIndividual:
        """Apply "num_mutation_functions" random mutation functions to the individual.

        Args:
            individual (BoardIndividual): The individual to mutate.

        Returns:
            BoardIndividual: The mutated individual.
        """
        probabilities = [
            mutation_function.probability for mutation_function in self.mutation_functions
        ]
        mutation_functions: List[MutationFunction] = self.random_state.choice(
            self.mutation_functions,
            size=self.num_mutation_functions,
            replace=False,
            p=probabilities,
        )
        for mutation_function in mutation_functions:
            individual = mutation_function(individual, *args, **kwargs)

        return individual


def get_mutation_strategy(
    mutation_strategy: str,
    mutation_functions: List[MutationFunction],
    *args: Any,
    **kwargs: Any,
) -> MutationStrategy:
    """Get the mutation strategy.

    Args:
        mutation_strategy (str): The strategy to use.
        mutation_functions (List[MutationFunction]): A **pointer** to the list of mutation functions.
            This means that the list of mutation functions can be modified externally and the changes
            will be reflected here.

    Returns:
        MutationStrategy: The mutation strategy.
    """
    if mutation_strategy == "all":
        return AllMutationFunctionsStrategy(mutation_functions)
    elif mutation_strategy == "one_random":
        return OneRandomMutationFunctionStrategy(mutation_functions)
    elif mutation_strategy == "n_random":
        return NRandomMutationFunctionsStrategy(mutation_functions, *args, **kwargs)
    else:
        raise ValueError(
            f"Invalid mutation strategy '{mutation_strategy}', must be one of ['all', 'one_random', 'n_random']"
        )


class Mutator:
    def __init__(
        self,
        mutation_strategy: str = "all",
        num_mutation_functions: Optional[int] = None,
        _random_state: Optional[np.random.Generator] = None,
    ):
        """A class for applying mutation functions to individuals.

        Args:
            mutation_strategy (str, optional): The strategy used to apply the mutation functions. Must be one of
                ["all", "one_random", "n_random"] where "all" applies all mutation functions, "one_random" applies one
                random mutation function, and "n_random" applies "num_mutation_functions" random mutation functions.
                Defaults to "all".
            num_mutation_functions (Optional[int], optional): The number of mutation functions to apply if the
                mutation strategy is "n_random". Defaults to None.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
        """
        self.num_mutation_functions = num_mutation_functions
        if mutation_strategy == "n_random":
            assert (
                self.num_mutation_functions is not None
            ), "Must specify the number of mutation functions to apply if the mutation strategy is 'n_random'"

        self.random_state = get_random_state(_random_state)
        self.mutation_functions: List[MutationFunction] = []
        self.mutation_strategy = get_mutation_strategy(
            mutation_strategy, self.mutation_functions, num_mutation_functions
        )
        self.global_probability: Optional[float] = None

    def register_mutation_function(
        self,
        functions: Union[
            Callable[[chess.Board, Any], chess.Board],
            List[Callable[[chess.Board, Any], chess.Board]],
        ],
        probability: float = 1.0,
        retries: int = 0,
        check_game_not_over: bool = False,
        clear_fitness_values: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Registers one or several mutation functions.

        Args:
            functions (Callable[[chess.Board, Any], chess.Board]): One or multiple mutation functions.
            retries (int, optional): The number of times to retry the mutation if the board is invalid. Defaults to 0.
            clear_fitness_values (bool, optional): Whether to clear the fitness values of the mutated individual.
                Defaults to False.
            *args: Default arguments to pass to the mutate function.
            **kwargs: Default keyword arguments to pass to the mutate function.
        """
        if not isinstance(functions, list):
            functions = [functions]

        for function in functions:
            self.mutation_functions.append(
                MutationFunction(
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

    def set_global_probability(self, probability: float) -> None:
        """Sets the probability of all mutation functions.

        Args:
            probability (float): The probability to set.
        """
        self.global_probability = probability
        for mutation_function in self.mutation_functions:
            mutation_function.probability = probability

    def __call__(self, individual: Individual, *args: Any, **kwargs: Any) -> Individual:
        """Mutates an individual.

        Args:
            individual (Individual): The individual to mutate.
            *args: Arguments to pass to the mutation functions.
            **kwargs: Keyword arguments to pass to the mutation functions.

        Returns:
            Individual: The mutated individual.
        """
        return self.mutation_strategy(individual, *args, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    board = chess.Board("8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118")
    print(board, "\n")
    mutate_flip_board(board, 1.0)
    print(board)
