import asyncio
import datetime
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.svg
import imgkit
import matplotlib
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import numpy as np


def get_task_result_handler(
    logger: logging.Logger,
    message: str,
    message_args: Tuple[Any, ...] = (),
) -> Any:
    def handle_task_result(
        task: asyncio.Task,
        *,
        logger: logging.Logger,
        message: str,
        message_args: Tuple[Any, ...] = (),
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Task cancellation should not be logged as an error.
        # Ad the pylint ignore: we want to handle all exceptions here so that the result of the task
        # is properly logged. There is no point re-raising the exception in this callback.
        except Exception:  # pylint: disable=broad-except
            logger.exception(message, *message_args)

    return lambda task: handle_task_result(
        task, logger=logger, message=message, message_args=message_args
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

    parsing_failed = False

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
        elif value == "nan":
            parsing_failed = True
        else:
            try:
                value = float(value)
            except ValueError:
                parsing_failed = True
                raise ValueError(f"Can't parse value {value}")

        if data_index in names:
            name = names[data_index]

        result[name] = value
    return result, parsing_failed


class MoveStat:
    def __init__(self, info: Dict[str, Any]):
        info_str = info["string"]
        info_str = "move:" + info_str
        info_str = info_str.replace(" N:", "(N:")
        if "(T)" in info_str:
            info_str = info_str.replace("(T)", "(T:1)")
        if "(L)" in info_str:
            info_str = info_str.replace("(L)", "(L:1)")
        if "(D)" in info_str:
            info_str = info_str.replace("(D)", "(D:1)")
        if "(W)" in info_str:
            info_str = info_str.replace("(W)", "(W:1)")

        parse_dic, parsing_failed = parse_stats(info_str, drop_indices=[1, 3], names={0: "move"})
        self.parsing_failed = parsing_failed
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

        parse_dic, parsing_failed = parse_stats(
            info_str, drop_indices=[1, 3], names={0: "position"}
        )
        self.parsing_failed = parsing_failed
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


def q2cp(q_value: float) -> float:
    return chess.engine.Cp(round(111.714640912 * np.tan(1.5620688421 * q_value)))


def cp2q(cp_value: float) -> float:
    return np.arctan(cp_value / 111.714640912) / 1.5620688421


def fen_to_file_name(fen: str, suffix: str = ""):
    fen = fen.replace("/", "|")
    fen = fen.replace(" ", "_")
    return fen + suffix


def plot_board(
    board: chess.Board,
    title: str = "",
    fen: str = "",
    fontsize: int = 22,
    save: bool = True,
    show: bool = False,
    save_path: Union[str, Path] = "",
) -> None:
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

    if save:
        if save_path == "":
            time_now = str(datetime.datetime.now())
            time_now = time_now.replace(" ", "_")
            save_path = Path(f"board_{time_now}.png")

        save_path = Path(save_path)
        save_path.absolute().parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()


if __name__ == "__main__":
    fen = "rnbqkbnr/pppp1ppp/8/3Pp3/5B2/6N1/PPP1PPPP/RNBQK2R w Kq e6 0 1"
    fen = "8/1p3r2/P1pK4/P3Pnp1/PpnR3N/Q4bqB/p2Bp3/Nk6 w - - 70 50"
    board = chess.Board(fen)

    plot_board(board=board, title="value 1: 0.99, value2: -0.99", fen=fen, fontsize=14)
