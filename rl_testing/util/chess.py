import datetime
import io
from pathlib import Path
from typing import Callable, List, Tuple, Union

import chess
import chess.engine
import chess.svg
import imgkit
import matplotlib
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import numpy as np

AGGRESSIVE_VALIDATION = True


VALUE_MATE_IN_MAX_PLY = 31754
NormalizeToPawnValue = 361
VALUE_MATE = 32000

NORMALIZED_SCORE_MAX = VALUE_MATE_IN_MAX_PLY * 100 / NormalizeToPawnValue


def extract_ply(fen: str) -> int:
    _, turn, _, _, _, fullmove = fen.split(" ")
    blacksturn = turn == "b"
    return (int(fullmove) - 1) * 2 + blacksturn


def win_rate(score: float, ply: int) -> int:
    """This is a reimplementation of the Stockfish win rate function in C++. The original
    version can be found here: https://github.com/official-stockfish/Stockfish/blob/master/src/uci.cpp#L202
    (Stockfish release 15.1)

    Args:
        score (float): The Stockfish score in centipawns.
        ply (int): The number of plies.

    Returns:
        int: The win rate in per mille units rounded to the nearest value.
    """

    # The model only captures up to 240 plies, so limit the input and then rescale
    m = min(240, ply) / 64.0

    # The coefficients of a third-order polynomial fit is based on the fishtest data
    # for two parameters that need to transform eval to the argument of a logistic
    # function.
    as_list = [-0.58270499, 2.68512549, 15.24638015, 344.49745382]
    bs_list = [-2.65734562, 15.96509799, -20.69040836, 73.61029937]

    # Enforce that NormalizeToPawnValue corresponds to a 50% win rate at ply 64
    assert NormalizeToPawnValue == int(as_list[0] + as_list[1] + as_list[2] + as_list[3])

    a = (((as_list[0] * m + as_list[1]) * m + as_list[2]) * m) + as_list[3]
    b = (((bs_list[0] * m + bs_list[1]) * m + bs_list[2]) * m) + bs_list[3]

    # Transform the eval to centipawns with limited range
    x = np.clip(score, -4000.0, 4000.0)

    # Return the win rate in per mille units rounded to the nearest value
    return int(0.5 + 1000 / (1 + np.exp((a - x) / b)))


def stockfish_cp_to_wdl(score: int, ply: int) -> Tuple[int, int, int]:
    """
    Converts a Stockfish score from centipawns to WDL.

    Args:
        score (float): The Stockfish score in centipawns.
        ply (int): The number of plies. Defined to be
            '(full_move_number - 1) * 2 + blacksturn' where
            'blacksturn' is 1 if it is Black's turn and 0
            otherwise

    Returns:
        Tuple[int, int, int]: The WDL score.
    """
    # Undoing the Stockfish normalization
    if abs(score) <= NORMALIZED_SCORE_MAX:
        score = (score * NormalizeToPawnValue) / 100

    wdl_w = win_rate(score, ply)
    wdl_l = win_rate(-score, ply)
    wdl_d = 1000 - wdl_w - wdl_l

    return wdl_w, wdl_d, wdl_l


def stockfish_cp_to_leela_q(score: int, ply: int) -> float:
    """Convert a Stockfish centipawn score to a Leela Q value.

    Args:
        score (int): The Stockfish score in centipawns.
        ply (int):

    Returns:
        float: _description_
    """
    win, draw, loss = stockfish_cp_to_wdl(score, ply)

    # Normalize to 0-1
    win, draw, loss = win / 1000, draw / 1000, loss / 1000

    # Convert to Leela Q
    return win - loss


def q2cp(q_value: float) -> float:
    return chess.engine.Cp(round(90 * np.tan(1.5637541897 * q_value)))


def cp2q(cp_value: float) -> float:
    return np.arctan(cp_value / 90) / 1.5637541897


def fen_to_file_name(fen: str, suffix: str = ""):
    fen = fen.replace("/", "|")
    fen = fen.replace(" ", "_")
    return fen + suffix


def remove_pawns(board: chess.Board) -> Union[chess.Board, str]:
    piece_map = board.piece_map()
    positions = list(piece_map.keys())
    for position in positions:
        if piece_map[position].symbol() in ["P", "p"]:
            del piece_map[position]

    new_board = chess.Board()
    new_board.set_piece_map(piece_map)
    new_board.turn = board.turn

    if not is_really_valid(new_board):
        return "failed"

    return new_board


def has_undefended_attacked_pieces(board: chess.Board) -> bool:
    """Checks whether a given board contains pieces which are attacked by the opponent
    but not defended by any of the player's pieces.

    Args:
        board (chess.Board): The board to check.

    Returns:
        bool: True if the board contains free pieces, False otherwise.
    """
    for square, piece in board.piece_map().items():
        if chess.piece_name(piece.piece_type) == "king":
            continue
        if board.is_attacked_by(not piece.color, square) and not board.is_attacked_by(
            piece.color, square
        ):
            return True
    return False


def apply_transformation(board: chess.Board, transformation: Callable) -> chess.Board:
    if transformation == "mirror":
        return board.mirror()

    return board.transform(transformation)


def rotate_90_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return chess.flip_vertical(chess.flip_diagonal(board))


def rotate_180_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return rotate_90_clockwise(rotate_90_clockwise(board))


def rotate_270_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return rotate_180_clockwise(rotate_90_clockwise(board))


def is_really_valid(board: chess.Board) -> bool:
    check_squares = list(board.checkers())
    if len(check_squares) > 2:
        return False

    if AGGRESSIVE_VALIDATION and len(check_squares) == 2:
        return False

    elif len(check_squares) == 2:
        if (
            board.piece_at(check_squares[0]).piece_type
            == board.piece_at(check_squares[1]).piece_type
        ):
            return False
        symbol1 = board.piece_at(check_squares[0]).symbol().lower()
        symbol2 = board.piece_at(check_squares[1]).symbol().lower()
        symbols = "".join([symbol1, symbol2])

        if "p" in symbols:
            return False

    return board.is_valid()


def transform_board_to_png(board: chess.Board, plot_size: int = 800, **kwargs) -> np.ndarray:
    # Get the XML representation of an SVG image of the board
    svg = chess.svg.board(board, size=plot_size, **kwargs)

    # Convert the XML representation into an SVG image
    transformed = imgkit.from_string(svg, output_path=False, options={"format": "png"})

    # Trick matplotlib into thinking that transformed is actually a file
    with io.BytesIO(transformed) as image_bytes:
        # Read the image as if it was a file
        im = mimage.imread(image_bytes)

    return im


def plot_board(
    board: chess.Board,
    title: str = "",
    x_label: str = "",
    fontsize: int = 22,
    plot_size: int = 800,
    save: bool = True,
    show: bool = False,
    save_path: Union[str, Path] = "",
    close_plot: bool = True,
    **kwargs,
) -> None:
    margin = 15

    im = transform_board_to_png(board, plot_size=plot_size, **kwargs)

    # Plot the image
    plt.imshow(im)

    # Change the font size
    font = {"size": fontsize}
    matplotlib.rc("font", **font)

    # Make the axes invisible
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    plt.xlim(0, plot_size + margin)
    plt.ylim(plot_size + margin, 0)

    if x_label:
        font["size"] = fontsize - 3
        plt.xlabel(x_label, fontdict=font, loc="left")

    # x_axis.set_visible(False)
    plt.xticks([])

    plt.title(title, pad=10)

    if save:
        if save_path == "":
            time_now = str(datetime.datetime.now())
            time_now = time_now.replace(" ", "_")
            save_path = Path(f"board_{time_now}.png")

        save_path = Path(save_path)
        save_path.absolute().parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")

    if show:
        plt.show()

    if close_plot:
        plt.close()


def plot_two_boards(
    board1: chess.Board,
    board2: chess.Board,
    arrows1: List[chess.svg.Arrow] = [],
    arrows2: List[chess.svg.Arrow] = [],
    title1: str = "",
    title2: str = "",
    x_label1: str = "",
    x_label2: str = "",
    fontsize: int = 22,
    plot_size: int = 800,
    save: bool = True,
    show: bool = False,
    save_path: Union[str, Path] = "",
    **kwargs,
):
    if save:
        assert save_path != "", "If save is True, save_path must be specified!"
    # margin = 15

    # Create the plot
    # figure, (ax1, ax2) = plt.subplots(1, 2)

    # Build the images
    # ax1.imshow(transform_board_to_png(board1, plot_size=plot_size, **kwargs))
    # ax2.imshow(transform_board_to_png(board2, plot_size=plot_size, **kwargs))

    plt.subplot(1, 2, 1)
    plot_board(
        board1,
        title=title1,
        x_label=x_label1,
        fontsize=fontsize,
        plot_size=plot_size,
        save=False,
        show=False,
        close_plot=False,
        arrows=arrows1,
        **kwargs,
    )

    plt.subplot(1, 2, 2)
    plot_board(
        board2,
        title=title2,
        x_label=x_label2,
        fontsize=fontsize,
        plot_size=plot_size,
        save=False,
        show=False,
        close_plot=False,
        arrows=arrows2,
        **kwargs,
    )

    if show:
        plt.show()

    if save:
        save_path = Path(save_path)
        save_path.absolute().parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    board1 = chess.Board("r4r1k/1pp3pp/p1pb1q2/4Nb2/2Q2B2/3P4/PPP2P2/RN3RK1 w - - 3 17")

    plot_board(board1, title="Board 1", save_path="board1.png", x_label="Evaluation: -0.79")
