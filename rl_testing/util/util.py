import io
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.svg
import imgkit
import matplotlib
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import numpy as np

FILES = "abcdefgh"
RANKS = list(range(1, 9))

CHESS_FIELDS = list(product(FILES, RANKS))

CHESS_PIECES_NON_ESSENTIAL = "RNBQBNRPPPPPPPPrnbqbnrpppppppp"

CASTLING_WHITE_KING_SIDE_REQUIRED = [(("e", 1), "K"), (("h", 1), "R")]
CASTLING_WHITE_KING_SIDE_FORBIDDEN = [("f", 1), ("g", 1)]

CASTLING_WHITE_QUEEN_SIDE_REQUIRED = [(("e", 1), "K"), (("a", 1), "R")]
CASTLING_WHITE_QUEEN_SIDE_FORBIDDEN = [("b", 1), ("c", 1), ("d", 1)]

CASTLING_BLACK_KING_SIDE_REQUIRED = [(("e", 8), "k"), (("h", 8), "r")]
CASTLING_BLACK_KING_SIDE_FORBIDDEN = [("f", 8), ("g", 8)]

CASTLING_BLACK_QUEEN_SIDE_REQUIRED = [(("e", 8), "k"), (("a", 8), "r")]
CASTLING_BLACK_QUEEN_SIDE_FORBIDDEN = [("b", 8), ("c", 8), ("d", 8)]


def random_board_position_fen(num_pieces: int) -> str:
    # Choose the 'num_pieces' chess pieces
    chess_pieces = "Kk" + "".join(
        np.random.choice(list(CHESS_PIECES_NON_ESSENTIAL), num_pieces, replace=False)
    )

    # Choose the position of the pieces
    chess_positions_idx = np.random.choice(
        list(range(len(CHESS_FIELDS))), num_pieces + 2, replace=False
    )
    chess_positions = [CHESS_FIELDS[i] for i in chess_positions_idx]

    # Map the positions to numerical values for convenience
    position_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

    # Create a raw board which will be processed into the real board
    board_raw = [
        (
            position_map[chess_positions[i][0]],
            8 - chess_positions[i][1],
            chess_pieces[i],
        )
        for i in range(len(chess_pieces))
    ]

    # Create a list of positions and pieces
    board_list = list(zip(chess_positions, chess_pieces))

    # Prepare a auxiliary datastructure to convert the board to fen
    position_array = [[] for i in range(8)]
    for file, rank, piece in board_raw:
        position_array[rank].append((file, piece))

    for i in range(8):
        position_array[i] = sorted(position_array[i])

    # Convert the auxiliatry datastructure to fen
    fen_position = ""
    for rank in range(8):
        old_rank = -1
        for rank, piece in position_array[rank]:
            if rank - 1 > old_rank:
                fen_position += str(rank - 1 - old_rank)
            old_rank = rank
            fen_position += piece

        if 8 - 1 > old_rank:
            fen_position += str(8 - 1 - old_rank)
        fen_position += "/"

    fen_position = fen_position[:-1]

    castling_right = ""
    if all([piece in board_list for piece in CASTLING_WHITE_KING_SIDE_REQUIRED]):
        castling_right += "K"
    if all([piece in board_list for piece in CASTLING_WHITE_QUEEN_SIDE_REQUIRED]):
        castling_right += "Q"
    if all([piece in board_list for piece in CASTLING_BLACK_KING_SIDE_REQUIRED]):
        castling_right += "k"
    if all([piece in board_list for piece in CASTLING_BLACK_QUEEN_SIDE_REQUIRED]):
        castling_right += "q"

    if castling_right != "":
        indices = np.random.choice(
            list(range(len(castling_right))), np.random.randint(1, len(castling_right) + 1)
        )
        castling_right = "".join(castling_right[i] for i in indices)

    else:
        castling_right = "-"

    color_to_move = np.random.choice(["w", "b"])

    # Find possible en-passant candidates
    en_passant_candidates = list(
        filter(
            lambda pos: (pos[0][0], pos[1], color_to_move) in [("c", "P", "b"), ("f", "p", "w")],
            board_list,
        )
    )

    if en_passant_candidates:
        en_passant_moves = [e[0][0] + str(e[0][1]) for e in en_passant_candidates]
        # Compute the probability that the last move was a pawn move leading to an
        # en-passant opportunity
        if np.random.rand() < 0.5:
            en_passant = np.random.choice(en_passant_moves)
        else:
            en_passant = "-"
    else:
        en_passant = "-"

    half_move = np.random.randint(0, 76)
    full_move = np.random.randint(0, 200)

    fen = (
        fen_position
        + " "
        + color_to_move
        + " "
        + castling_right
        + " "
        + en_passant
        + " "
        + str(half_move)
        + " "
        + str(full_move)
    )

    return fen


def random_valid_board(
    num_pieces: int, max_attempts_per_position: int = 100
) -> Union[chess.Board, str]:
    for attempt in range(max_attempts_per_position):
        fen = random_board_position_fen(num_pieces=num_pieces)
        try:
            board = chess.Board(fen=fen)
            if board.is_valid() and (not board.is_checkmate()) and (not board.is_stalemate()):
                return board
        except ValueError as ve:
            print(ve)

    return "failed"


def contains_move_stat(info: Dict[str, Any]) -> bool:
    return "string" in info and "node" not in info["string"]


def contains_position_stat(info: Dict[str, Any]) -> bool:
    return "string" in info and "node" in info["string"]


def parse_stats(
    info_str: str,
    drop_indices: Optional[List[int]] = None,
    names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    if drop_indices is None:
        drop_indices = []
    if names is None:
        names = {}
    result = {}
    result_list = info_str.split("(")

    for data_index, data in enumerate(result_list):
        if data_index in drop_indices:
            continue
        if ")" in data:
            index = data.index(")")
            data = data[:index]
        name, value = data.split(":")
        name, value = name.strip(), value.strip()

        if value.endswith("%"):
            value = float(value[:-1]) / 100
        elif "-.-" in value:
            value = None
        elif name == "move":
            value = chess.Move.from_uci(value)
        elif name == "position":
            assert value == "node"
        else:
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"Can't parse value {value}")

        if data_index in names:
            name = names[data_index]

        result[name] = value
    return result


class MoveStat:
    def __init__(self, info: Dict[str, Any]):
        info_str = info["string"]
        info_str = "move:" + info_str
        info_str = info_str.replace(" N:", "(N:")
        if "(T)" in info_str:
            info_str = info_str.replace("(T)", "(T:1)")

        parse_dic = parse_stats(info_str, drop_indices=[1, 3], names={0: "move"})
        self.move = parse_dic["move"]
        self.N = parse_dic["N"]
        self.P = parse_dic["P"]
        self.WL = parse_dic["WL"]
        self.D = parse_dic["D"]
        self.M = parse_dic["M"]
        self.Q = parse_dic["Q"]
        self.U = parse_dic["U"]
        self.S = parse_dic["S"]
        self.V = parse_dic["V"]

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return str(self.__repr__())


class PositionStat:
    def __init__(self, info: Dict[str, Any]):
        info_str = info["string"]
        info_str = "position:" + info_str
        info_str = info_str.replace(" N:", "(N:")
        if "(T)" in info_str:
            info_str = info_str.replace("(T)", "(T:1)")

        parse_dic = parse_stats(info_str, drop_indices=[1, 3], names={0: "position"})
        self.N = parse_dic["N"]
        self.P = parse_dic["P"]
        self.WL = parse_dic["WL"]
        self.D = parse_dic["D"]
        self.M = parse_dic["M"]
        self.Q = parse_dic["Q"]
        self.V = parse_dic["V"]

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return str(self.__repr__())


def parse_info(
    info: Dict[str, Any], raise_exception: bool = False
) -> Optional[Union[MoveStat, PositionStat]]:
    if contains_move_stat(info):
        return MoveStat(info)
    elif contains_position_stat(info):
        return PositionStat(info)
    else:
        if raise_exception:
            raise ValueError("The provided dictionary cannot be parsed!")


def plot_board(board: chess.Board, title: str = "", fen: str = "", fontsize: int = 22) -> None:
    # Get the XML representation of an SVG image of the board
    svg = chess.svg.board(board)

    # Convert the XML representation into an SVG image
    transformed = imgkit.from_string(svg, output_path=False, options={"format": "png"})

    # Trick matplotlib into thinking that transformed is actually a file
    with io.BytesIO(transformed) as image_bytes:

        # Read the image as if it was a file
        im = mimage.imread(image_bytes)

    # Plot the image
    plt.imshow(im)

    # Change the font size
    font = {"size": fontsize}
    matplotlib.rc("font", **font)

    # Make the axes invisible
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    if fen:
        font["size"] = fontsize - 4
        plt.xlabel(fen, fontdict=font)

    # x_axis.set_visible(False)
    plt.xticks([])

    plt.title(title, pad=10)

    plt.show()


if __name__ == "__main__":
    fen = "rnbqkbnr/pppp1ppp/8/3Pp3/5B2/6N1/PPP1PPPP/RNBQK2R w Kq e6 0 1"
    fen = "8/1p3r2/P1pK4/P3Pnp1/PpnR3N/Q4bqB/p2Bp3/Nk6 w - - 70 50"
    board = chess.Board(fen)

    plot_board(board=board)

    ENGINE_CONFIG = {
        "Backend": "cuda-auto",
        "WeightsFile": "/home/flurilu/Software/leelachesszero/lc0/build/release/weights/"
        "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42",
        "VerboseMoveStats": "true",
        "SmartPruningFactor": "0",
        "Threads": "1",
        "TaskWorkers": "0",
        "MinibatchSize": "1",
        "MaxPrefetch": "0",
        "NNCacheSize": "1",
    }

    # for key, val in ENGINE_CONFIG.items():
    #    print(f"setoption name {key} value {val}")

    # for i in range(1000):
    #    # print(random_board_position_fen(num_pieces=20))
    #    print(random_valid_board(num_pieces=20, max_attempts_per_position=10000).fen())
