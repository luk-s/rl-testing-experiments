import argparse
from pathlib import Path
from typing import Optional
from rl_testing.distributed.distributed_queue_manager import (
    default_address,
    default_port,
    default_password,
)
from rl_testing.distributed.queue_utils import QueueInterface
from rl_testing.distributed.worker import PlayObject
from rl_testing.data_generators import get_data_generator
from rl_testing.config_parsers import get_data_generator_config

import chess.pgn
from datetime import datetime
import logging

# How an example run could look like:
# python engine_duel.py --port1 50000 --port2 50001 --data_config_name openings_move8_1.ini --required_engine_config_name1 leela_local_400_nodes.ini --required_engine_config_name2 stockfish_local_30000.ini
# ...
# python worker.py --port 50000 --engine_config_name leela_local_400_nodes.ini
# ...
# python worker.py --port 50001 --engine_config_name stockfish_local_30000.ini
# ...

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
    address1: str = default_address,
    port1: str = default_port,
    password1: str = default_password,
    required_config_name2: Optional[str] = None,
    address2: str = default_address,
    port2: str = default_port,
    password2: str = default_password,
) -> None:
    """Starts an engine duel between two engines.

    Args:
        data_generator (BoardGenerator): The data generator to use.
        num_openings (int): The number of openings to play.
        address1 (str, optional): The address of the first engine. Defaults to default_address.
        port1 (str, optional): The port of the first engine. Defaults to default_port.
        password1 (str, optional): The password of the first engine. Defaults to default_password.
        address2 (str, optional): The address of the second engine. Defaults to default_address.
        port2 (str, optional): The port of the second engine. Defaults to default_port.
        password2 (str, optional): The password of the second engine. Defaults to default_password.
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
        address=address1,
        port=port1,
        password=password1,
        required_engine_config=required_config_name1,
    )

    engine2 = QueueInterface(
        address=address2,
        port=port2,
        password=password2,
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

    # Start the engine duel
    engine_duel(
        data_generator_config_name=args.data_config_name,
        num_openings=args.num_openings,
        required_config_name1=args.required_config_name1,
        address1=args.address1,
        port1=args.port1,
        password1=args.password1,
        required_config_name2=args.required_config_name2,
        address2=args.address2,
        port2=args.port2,
        password2=args.password2,
    )
