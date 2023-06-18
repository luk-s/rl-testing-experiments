import argparse
from pathlib import Path
from typing import Optional
from rl_testing.distributed.distributed_queue_manager import (
    default_address,
    default_port,
    default_password,
)
from rl_testing.distributed.queue_utils import QueueInterface, SocketAddress, EmptySocketAddress
from rl_testing.distributed.worker import PlayObject
from rl_testing.data_generators import get_data_generator
from rl_testing.config_parsers import get_data_generator_config

import chess.pgn
from datetime import datetime
import logging

RESULT_DIR = Path(__file__).parent / Path("results/engine_duel")
CONFIG_FOLDER_PATH = Path(__file__).parent.absolute() / Path("configs/data_generator_configs/")


def play_game(
    start_position: str,
    engine_white: QueueInterface,
    engine_black: QueueInterface,
    round: Optional[int] = None,
) -> chess.pgn.Game:
    """Plays a game between two engines and returns the game.

    Args:
        start_position (str): The start position of the game in FEN notation.
        engine_white (QueueInterface): An interface to communicate with the engine
            playing as white.
        engine_black (QueueInterface): An interface to communicate with the engine
            playing as black.
        round (Optional[int], optional): The round number of the game. Defaults to None.

    Returns:
        chess.pgn.Game: The game played.
    """
    logging.info(
        f"Playing game {round} with start position {start_position} and engines"
        f" {engine_white.engine_config_name} and {engine_black.engine_config_name}"
    )

    # Prepare the game
    game = chess.pgn.Game()
    game.headers["Event"] = "Engine Duel"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = round if round is not None else "?"
    game.headers["White"] = engine_white.engine_config_name
    game.headers["Black"] = engine_black.engine_config_name
    game.headers["Result"] = "*"
    game.headers["FEN"] = start_position

    board = chess.Board(start_position)

    move_number = 0

    # Initialize the current game node
    node = game

    while not board.is_game_over(claim_draw=True):
        logging.info(f"Current position: {board.fen(en_passant='fen')}")

        # Send the current position to the engine
        engine = engine_white if board.turn == chess.WHITE else engine_black
        engine.send(
            PlayObject(
                board=board.copy(),
                new_game=move_number == 0,
                ponder=False,
            )
        )

        move_number += 1

        # Wait for the engine to respond
        result: PlayObject = engine.receive()

        # If the analysis was not successful, abort the game
        if result.best_move is None or result.best_move == "invalid":
            game.headers["Result"] = "1/2-1/2"
            game.headers["Termination"] = "Abandoned"
            break

        move = result.best_move

        logging.info(f" Engine {engine.engine_config_name} played move {move}")

        node = node.add_variation(
            move,
        )
        board.push(move)
    else:
        # If the game finished normally, set the result
        game.headers["Result"] = board.result(claim_draw=True)

    logging.info(f"Game finished with result {game.headers['Result']}")
    if "Termination" in game.headers:
        logging.info(f"Termination: {game.headers['Termination']}")

    return game


def engine_duel(
    data_generator_config_name: str,
    num_openings: int,
    required_config_name1: Optional[str] = None,
    required_config_name2: Optional[str] = None,
    socket_address1: SocketAddress = EmptySocketAddress,
    socket_address2: SocketAddress = EmptySocketAddress,
) -> None:
    """Starts an engine duel between two engines.

    Args:
        data_generator (BoardGenerator): The data generator to use.
        num_openings (int): The number of openings to play.
        required_config_name1 (Optional[str], optional): The name of the engine config
            required by the first engine. Defaults to None.
        required_config_name2 (Optional[str], optional): The name of the engine config
            required by the second engine. Defaults to None.
        socket_address1 (SocketAddress, optional): An object that contains address information
            about the first queue. Defaults to EmptySocketAddress.
        socket_address2 (SocketAddress, optional): An object that contains address information
            about the second queue. Defaults to EmptySocketAddress.
    """
    # Build the result file path
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULT_DIR / Path(
        f"ENGINE1_{required_config_name1}_ENGINE2_{required_config_name2}_"
        f"{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}.pgn"
    )

    # Build the data generator
    data_generator_config = get_data_generator_config(
        data_generator_config_name,
        config_folder_path=CONFIG_FOLDER_PATH,
    )
    data_generator = get_data_generator(data_generator_config)

    # Set up the distributed queues
    engine1 = QueueInterface(
        **socket_address1.to_dict(),
        required_engine_config=required_config_name1,
    )

    engine2 = QueueInterface(
        **socket_address2.to_dict(),
        required_engine_config=required_config_name2,
    )

    # Play the games
    for i in range(num_openings):
        # Get the next opening
        board = data_generator.next()

        # Play the game
        game1 = play_game(
            start_position=board.fen(en_passant="fen"),
            engine_white=engine1,
            engine_black=engine2,
            round=i + 1,
        )

        # Play the reverse game
        game2 = play_game(
            start_position=board.fen(en_passant="fen"),
            engine_white=engine2,
            engine_black=engine1,
            round=i + 1,
        )

        # Save the game
        print(game1, file=open(result_file, "a"), end="\n\n")
        print(game2, file=open(result_file, "a"), end="\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    parser.add_argument("--data_config_name",     type=str, default="openings_move8_1.ini") # noqa
    parser.add_argument("--num_openings",         type=int, default=1) # noqa
    parser.add_argument("--required_config_name1",type=str, default="leela_local_400_nodes.ini") # noqa
    parser.add_argument("--address1",             type=str, default=default_address) # noqa
    parser.add_argument("--port1",                type=int, default=default_port) # noqa
    parser.add_argument("--password1",            type=str, default=default_password) # noqa
    parser.add_argument("--required_config_name2",type=str, default="stockfish_local_30000.ini") # noqa
    parser.add_argument("--address2",             type=str, default=default_address) # noqa
    parser.add_argument("--port2",                type=int, default=default_port) # noqa
    parser.add_argument("--password2",            type=str, default=default_password) # noqa
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################

    args = parser.parse_args()

    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()

    socket_address1 = SocketAddress(args.address1, args.port1, args.password1)
    socket_address2 = SocketAddress(args.address2, args.port2, args.password2)

    # Start the engine duel
    engine_duel(
        data_generator_config_name=args.data_config_name,
        num_openings=args.num_openings,
        required_config_name1=args.required_config_name1,
        required_config_name2=args.required_config_name2,
        socket_address1=socket_address1,
        socket_address2=socket_address2,
    )
