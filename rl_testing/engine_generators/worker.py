import argparse
import asyncio
import logging
import queue
from multiprocessing import current_process
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chess
import chess.engine

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.engine_generators.distributed_queue_manager import (
    connect_to_manager,
    password,
)
from rl_testing.util.chess import cp2q
from rl_testing.util.util import get_task_result_handler


class AnalysisObject:
    def __init__(self, fen: str):
        self.fen = fen
        self.score: Optional[float] = None


class TransformationAnalysisObject(AnalysisObject):
    def __init__(self, fen: str, base_fen: str, transformation_index: int):
        self.fen = fen
        self.base_fen = base_fen
        self.transformation_index = transformation_index
        self.score: Optional[float] = None


PLACEHOLDER_ANALYSIS_OBJECT = AnalysisObject(fen="")


async def analyze_position(
    search_limits: Dict[str, Any],
    engine_generator: EngineGenerator,
    network_name: Optional[str] = None,
    identifier_str: str = "",
) -> None:
    consumer_queue, producer_queue = connect_to_manager()
    # Authenticate the engine
    # current_process().authkey = password.encode("utf-8")

    board_counter = 1
    # Required to ensure that the engine doesn't use cached results from
    # previous analyses
    analysis_counter = 0

    # Initialize the engine
    if network_name is not None:
        engine_generator.set_network(network_name=network_name)
    engine = await engine_generator.get_initialized_engine()

    while True:
        # Fetch the next base board, the next transformed board, and the corresponding
        # transformation index
        print(f"[{identifier_str}] Before get")
        print(f"[{identifier_str}] Size: ", consumer_queue.qsize())
        analysis_object: AnalysisObject = consumer_queue.get()
        print("After get")
        current_board = chess.Board(analysis_object.fen)

        logging.info(
            f"[{identifier_str}] Analyzing board {board_counter}: "
            + current_board.fen(en_passant="fen")
        )
        try:
            # Analyze the board
            analysis_counter += 1
            info = await engine.analyse(
                current_board, chess.engine.Limit(**search_limits), game=analysis_counter
            )
            print(f"[{identifier_str}] after analyze of board {board_counter}")

        except chess.engine.EngineTerminatedError:
            if engine_generator is None:
                logging.info("Can't restart engine due to missing generator")
                raise

            # Try to kill the failed engine
            logging.info(f"[{identifier_str}] Trying to kill engine")
            engine_generator.kill_engine(engine=engine)

            # Try to restart the engine
            logging.info("Trying to restart engine")

            if network_name is not None:
                engine_generator.set_network(network_name=network_name)
            engine = await engine_generator.get_initialized_engine()

            # Add an error to the receiver queue
            analysis_object.score = "invalid"
        else:
            # Get the score of the board

            # Get the score of the most promising child board
            score_cp = info["score"].relative.score(mate_score=12780)
            score_q = cp2q(score_cp)
            analysis_object.score = score_q

        finally:
            # Add the board to the receiver queue
            producer_queue.put(analysis_object)
            print(f"[{identifier_str}] Marking task {board_counter} as done")
            consumer_queue.task_done()
            board_counter += 1


async def main(
    args: argparse.Namespace,
    logger: logging.Logger,
):
    # Create result directory
    config_folder_path = Path(__file__).absolute().parent.parent.parent / Path(
        "experiments/configs/engine_configs/"
    )

    # Build the engine generator
    engine_config = get_engine_config(
        config_name=args.engine_config_name, config_folder_path=config_folder_path
    )
    assert (
        "verbosemovestats" in engine_config.engine_config
        and engine_config.engine_config["verbosemovestats"]
    ), "VerboseMoveStats must be enabled in the engine config"
    engine_generator = get_engine_generator(engine_config)

    # Get the current process name
    process_name = current_process().name

    # Start the tasks
    analysis_task = asyncio.create_task(
        analyze_position(
            search_limits=engine_config.search_limits,
            engine_generator=engine_generator,
            network_name=args.network_name,
            identifier_str=f"ANALYSIS {process_name}",
        )
    )
    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    analysis_task.add_done_callback(handle_task_exception)

    # Wait for data generator task to finish
    await asyncio.wait([analysis_task])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    parser.add_argument("--seed",                  type=int, default=42)  # noqa
    parser.add_argument("--engine_config_name",    type=str, default="local_400_nodes.ini")  # noqa
    # parser.add_argument("--engine_config_name",    type=str, default="remote_400_nodes.ini")  # noqa
    # parser.add_argument("--num_positions",         type=int, default=100)  # noqa
    # parser.add_argument("--network_path",          type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa
    parser.add_argument("--network_name",          type=str,  default="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207")  # noqa
    # parser.add_argument("--network_path",          type=str,  default="T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2")  # noqa
    parser.add_argument("--queue_max_size",        type=int, default=100_000)  # noqa
    parser.add_argument("--result_subdir",         type=str, default="")  # noqa
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################
    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()

    # Parse command line arguments
    args = parser.parse_args()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        main(
            args=args,
            logger=logger,
        )
    )
