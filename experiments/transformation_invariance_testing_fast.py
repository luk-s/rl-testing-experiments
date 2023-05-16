import argparse
import asyncio
import logging
import os
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import chess
import chess.engine
import numpy as np
from chess import flip_anti_diagonal, flip_diagonal, flip_horizontal, flip_vertical

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.engine_generators.distributed_queue_manager import (
    QueueManager,
    address,
    connect_to_manager,
    password,
    port,
)
from rl_testing.engine_generators.worker import AnalysisObject
from rl_testing.util.chess import apply_transformation, cp2q
from rl_testing.util.chess import remove_pawns as remove_pawns_func
from rl_testing.util.chess import (
    rotate_90_clockwise,
    rotate_180_clockwise,
    rotate_270_clockwise,
)
from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import get_task_result_handler

RESULT_DIR = Path(__file__).parent / Path("results/transformation_testing")

transformation_dict = {
    "rot90": rotate_90_clockwise,
    "rot180": rotate_180_clockwise,
    "rot270": rotate_270_clockwise,
    "flip_diag": flip_diagonal,
    "flip_anti_diag": flip_anti_diagonal,
    "flip_hor": flip_horizontal,
    "flip_vert": flip_vertical,
    "mirror": "mirror",
}


class TransformationAnalysisObject(AnalysisObject):
    def __init__(self, fen: str, base_fen: str, transformation_index: int):
        self.fen = fen
        self.base_fen = base_fen
        self.transformation_index = transformation_index
        self.score: Optional[float] = None


class ReceiverCache:
    def __init__(self, queue: queue.Queue, num_transformations: int) -> None:
        assert isinstance(queue, queue.Queue)
        self.queue = queue
        self.num_transformations = num_transformations

        self.score_cache: Dict[str, List[Optional[float]]] = {}

    async def receive_data(self) -> List[Iterable[Any]]:
        # Receive data from queue
        while True:
            try:
                analysis_object: TransformationAnalysisObject = self.queue.get_nowait()
                base_fen = analysis_object.base_fen
                transform_index = analysis_object.transformation_index
                score = analysis_object.score
            except queue.Empty:
                await asyncio.sleep(delay=0.5)
            else:
                await asyncio.sleep(delay=0.1)
                break

        # The boards might not arrive in the correct order due to the asynchronous nature of
        # the program. Therefore, we need to cache the boards and scores until we have all
        # of them.

        if base_fen in self.score_cache:
            self.score_cache[base_fen][transform_index] = score
        else:
            self.score_cache[base_fen] = [None] * self.num_transformations
            self.score_cache[base_fen][transform_index] = score

        complete_data_tuples = []
        # Check if we have all the data for this board
        if all([element is not None for element in self.score_cache[base_fen]]):
            # We have all the data for this board
            complete_data_tuples.append((base_fen, self.score_cache[base_fen]))
            del self.score_cache[base_fen]

        self.queue.task_done()

        return complete_data_tuples


async def create_positions(
    data_generator: BoardGenerator,
    transformation_functions: List[Callable[[chess.Bitboard], chess.Bitboard]],
    remove_pawns: bool = False,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    identifier_str: str = "",
) -> None:
    fen_cache = {}

    # Get the queues
    output_queue: queue.Queue
    output_queue, _ = connect_to_manager()

    # This website might be helpful: https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating # noqa: E501
    # Create random chess positions if necessary
    board_index = 1
    while board_index <= num_positions:

        # Create a random chess position
        board_candidate = data_generator.next()

        if board_candidate != "failed" and remove_pawns:
            board_candidate = remove_pawns_func(board_candidate)

        # Check if the generated position was valid
        if board_candidate != "failed" and board_candidate.fen() not in fen_cache:
            fen_cache[board_candidate.fen()] = True

            # Apply the transformations to the board
            transformed_boards = [board_candidate]
            for transformation_function in transformation_functions:
                transformed_boards.append(
                    apply_transformation(board_candidate, transformation_function)
                )

            fen = board_candidate.fen(en_passant="fen")

            logging.info(f"[{identifier_str}] Created base board {board_index + 1}: {fen}")
            for transformed_board in transformed_boards[1:]:
                fen = transformed_board.fen(en_passant="fen")
                logging.info(f"[{identifier_str}] Created transformed board: {fen}")

            for transform_index, transformed_board in enumerate(transformed_boards):
                analysis_object = TransformationAnalysisObject(
                    fen=transformed_board.fen(en_passant="fen"),
                    base_fen=board_candidate.fen(en_passant="fen"),
                    transformation_index=transform_index,
                )
                output_queue.put(analysis_object)

            await asyncio.sleep(delay=sleep_between_positions)

            board_index += 1


async def evaluate_candidates(
    num_transforms: int,
    file_path: Union[str, Path],
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    # Get the queues
    engine_queue: queue.Queue
    _, engine_queue = connect_to_manager()

    # Create a file to store the results
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    board_counter = 1

    # The order of the queues is important! The 'receive' function will return the data in the
    # same order as the queues are given to the initializer.
    receiver_cache = ReceiverCache(queue=engine_queue, num_transformations=num_transforms)

    with open(file_path, "a") as file:
        flush_every = 1000
        while True:
            # Fetch the next board and the corresponding scores from the queues
            complete_data_tuples = await receiver_cache.receive_data()

            # Iterate over the received data
            for fen, scores in complete_data_tuples:
                logging.info(f"[{identifier_str}] Saving board {board_counter}: " + fen)

                # Write the found adversarial example into a file
                result_str = f"{fen},"
                for score in scores:
                    # Add the score to the result string
                    result_str += f"{score},"

                    # Mark the element as processed
                    engine_queue.task_done()

                result_str = result_str[:-1] + "\n"

                # Write the result to the file
                file.write(result_str)

                if board_counter % flush_every == 0:
                    file.flush()
                    os.fsync(file.fileno())

                board_counter += 1


async def transformation_invariance_testing(
    data_generator: BoardGenerator,
    transformation_functions: List[Callable[[chess.Bitboard], chess.Bitboard]],
    *,
    result_file_path: Optional[Union[str, Path]] = None,
    remove_pawns: bool = False,
    num_positions: int = 1,
    sleep_after_get: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> None:

    if logger is None:
        logger = logging.getLogger(__name__)

    assert result_file_path is not None, "Result file path must be specified"

    # Set up the distributed queues
    engine_queue_in: queue.Queue = queue.Queue()
    engine_queue_out: queue.Queue = queue.Queue()

    def get_input_queue() -> queue.Queue:
        return engine_queue_in

    def get_output_queue() -> queue.Queue:
        return engine_queue_out

    # Initialize the input- and output queues
    QueueManager.register("input_queue", callable=get_input_queue)
    QueueManager.register("output_queue", callable=get_output_queue)

    net_manager = QueueManager(address=(address, port), authkey=password.encode("utf-8"))

    # Start the server
    net_manager.start()

    # Create all data processing tasks
    data_generator_task = asyncio.create_task(
        create_positions(
            data_generator=data_generator,
            transformation_functions=transformation_functions,
            remove_pawns=remove_pawns,
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            num_transforms=len(transformation_functions) + 1,
            file_path=result_file_path,
            sleep_after_get=sleep_after_get,
            identifier_str="CANDIDATE_EVALUATION",
        )
    )

    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    for task in [
        data_generator_task,
        candidate_evaluation_task,
    ]:
        task.add_done_callback(handle_task_exception)

    # Wait for data generator task to finish
    await asyncio.wait([data_generator_task])

    # Wait for data queues to become empty
    engine_queue_in.join()
    engine_queue_out.join()

    # Cancel all remaining tasks
    candidate_evaluation_task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # NETWORKS:
    # =========
    # strong and recent: "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    # from leela paper: "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    # strong recommended: "network_600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2" # noqa: E501
    # Weak local 1: "f21ee51844a7548c004a1689eacd8b4cd4c6150d6e03c732b211cf9963d076e1"
    # Weak local 2: "fbd5e1c049d5a46c098f0f7f12e79e3fb82a7a6cd1c9d1d0894d0aae2865826f"

    # fmt: off
    parser.add_argument("--seed",                           type=int, default=42)  # noqa
    parser.add_argument("--engine_config_name",             type=str, default="local_400_nodes.ini")  # noqa
    parser.add_argument("--data_config_name",               type=str, default="database.ini")  # noqa
    parser.add_argument("--remove_pawns",                   action="store_true")  # noqa
    parser.add_argument("--num_positions",                  type=int, default=1_000_000)  # noqa
    # parser.add_argument("--num_positions",                  type=int, default=100)  # noqa
    # parser.add_argument("--network_path",                   type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa
    parser.add_argument("--transformations",                type=str, default=["rot90", "rot180", "rot270", "flip_diag", "flip_anti_diag", "flip_hor", "flip_vert"], nargs="+",  # noqa
                                                            choices=["rot90", "rot180", "rot270", "flip_diag", "flip_anti_diag", "flip_hor", "flip_vert", "mirror"])  # noqa
    parser.add_argument("--result_subdir",                  type=str, default="")  # noqa
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

    # Set random seed
    np.random.seed(args.seed)

    # Create result directory
    config_folder_path = Path(__file__).parent.absolute() / Path("configs/engine_configs/")

    # Build the data generator
    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Extract the transformations
    transformation_functions = [
        transformation_dict[transformation_name] for transformation_name in args.transformations
    ]

    # Create results-file-name
    data_config_name = args.data_config_name[:-4]

    # Build the result file path
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)

    # Store current date and time as string
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")

    result_file_path = result_directory / Path(
        f"results_ENGINE_{args.engine_config_name}_DATA_{data_config_name}_{dt_string}.txt"
    )

    # Store the experiment configuration in the result file
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Store the transformation names in the result file
    with open(result_file_path, "a") as result_file:
        result_file.write("fen,original,")
        transformation_str = "".join(
            [f"{transformation}," for transformation in args.transformations]
        )
        result_file.write(f"{transformation_str[:-1]}\n")

    # Extract the boolean parameter
    remove_pawns = args.remove_pawns
    if remove_pawns is None:
        remove_pawns = False

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        transformation_invariance_testing(
            data_generator=data_generator,
            transformation_functions=transformation_functions,
            result_file_path=result_file_path,
            remove_pawns=remove_pawns,
            num_positions=args.num_positions,
            queue_max_size=args.queue_max_size,
            num_engine_workers=args.num_engine_workers,
            sleep_after_get=0.1,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
