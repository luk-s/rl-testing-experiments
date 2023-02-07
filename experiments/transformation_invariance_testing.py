import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import chess
import chess.engine
import numpy as np
from chess import flip_anti_diagonal, flip_diagonal, flip_horizontal, flip_vertical

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.util.chess import (
    remove_pawns,
    rotate_90_clockwise,
    rotate_180_clockwise,
    rotate_270_clockwise,
)
from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import cp2q, get_task_result_handler

RESULT_DIR = Path(__file__).parent / Path("results/transformation_testing")

transformation_dict = {
    "rot90": rotate_90_clockwise,
    "rot180": rotate_180_clockwise,
    "rot270": rotate_270_clockwise,
    "flip_diag": flip_diagonal,
    "flip_anti_diag": flip_anti_diagonal,
    "flip_hor": flip_horizontal,
    "flip_vert": flip_vertical,
}


class ReceiverCache:
    def __init__(self, queue: asyncio.Queue, num_transformations: int) -> None:
        assert isinstance(queue, asyncio.Queue)
        self.queue = queue
        self.num_transformations = num_transformations

        self.score_cache = {}

    async def receive_data(self) -> List[Iterable[Any]]:
        # Receive data from queue
        base_board, transform_index, score = await self.queue.get()

        assert isinstance(base_board, chess.Board)

        # The boards might not arrive in the correct order due to the asynchronous nature of
        # the program. Therefore, we need to cache the boards and scores until we have all
        # of them.
        fen = base_board.fen()

        if fen in self.score_cache:
            self.score_cache[fen][transform_index] = score
        else:
            self.score_cache[fen] = [None] * self.num_transformations
            self.score_cache[fen][transform_index] = score

        complete_data_tuples = []
        # Check if we have all the data for this board
        if all([element is not None for element in self.score_cache[fen]]):
            # We have all the data for this board
            complete_data_tuples.append((fen, self.score_cache[fen]))
            del self.score_cache[fen]

        return complete_data_tuples


async def create_positions(
    queues: List[asyncio.Queue],
    data_generator: BoardGenerator,
    transformation_functions: List[Callable[[chess.Bitboard], chess.Bitboard]],
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    identifier_str: str = "",
) -> None:
    fen_cache = {}

    # This website might be helpful: https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating # noqa: E501
    # Create random chess positions if necessary
    board_index = 1
    while board_index <= num_positions:

        # Create a random chess position
        board_candidate = data_generator.next()

        if board_candidate != "failed":
            board_candidate = remove_pawns(board_candidate)

        # Check if the generated position was valid
        if board_candidate != "failed" and board_candidate.fen() not in fen_cache:
            fen_cache[board_candidate.fen()] = True

            # Apply the transformations to the board
            transformed_boards = [board_candidate]
            for transformation_function in transformation_functions:
                transformed_boards.append(board_candidate.transform(transformation_function))

            fen = board_candidate.fen(en_passant="fen")

            logging.info(f"[{identifier_str}] Created base board {board_index + 1}: " f"{fen}")
            for transformed_board in transformed_boards[1:]:
                fen = transformed_board.fen(en_passant="fen")
                logging.info(f"[{identifier_str}] Created transformed board: " f"{fen}")

            for queue in queues:
                for transform_index, transformed_board in enumerate(transformed_boards):
                    await queue.put((board_candidate.copy(), transform_index, transformed_board))

            await asyncio.sleep(delay=sleep_between_positions)

            board_index += 1


async def analyze_position(
    consumer_queue: asyncio.Queue,
    producer_queue: asyncio.Queue,
    search_limits: Dict[str, Any],
    engine_generator: EngineGenerator,
    network_name: Optional[str] = None,
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
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
        base_board, transform_index, transformed_board = await consumer_queue.get()
        await asyncio.sleep(delay=sleep_after_get)

        logging.info(
            f"[{identifier_str}] Analyzing board {board_counter}: "
            + transformed_board.fen(en_passant="fen")
        )
        try:
            # Analyze the board
            analysis_counter += 1
            info = await engine.analyse(
                transformed_board,
                chess.engine.Limit(**search_limits),
                game=analysis_counter,
            )
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
            await producer_queue.put((base_board, transform_index, "invalid"))
        else:
            # Add the board to the receiver queue
            # The 12800 is used as maximum value because we use the q2cp function
            # to convert q_values to centipawns. This formula has values in
            # [-12800, 12800] for q_values in [-1, 1]
            await producer_queue.put(
                (
                    base_board,
                    transform_index,
                    cp2q(info["score"].relative.score(mate_score=12800)),
                )
            )
        finally:
            consumer_queue.task_done()
            board_counter += 1


async def evaluate_candidates(
    engine_queue: asyncio.Queue,
    num_transforms: int,
    file_path: Union[str, Path],
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    # Create a file to store the results
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    board_counter = 1

    # The order of the queues is important! The 'receive' function will return the data in the
    # same order as the queues are given to the initializer.
    receiver_cache = ReceiverCache(queue=engine_queue, num_transformations=num_transforms)

    with open(file_path, "a") as file:

        while True:
            # Fetch the next board and the corresponding scores from the queues
            complete_data_tuples = await receiver_cache.receive_data()

            # Iterate over the received data
            for fen, scores in complete_data_tuples:
                await asyncio.sleep(delay=sleep_after_get)

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

                board_counter += 1


async def transformation_invariance_testing(
    engine_generator: EngineGenerator,
    network_name: Optional[str],
    data_generator: BoardGenerator,
    transformation_functions: List[Callable[[chess.Bitboard], chess.Bitboard]],
    *,
    search_limits: Optional[Dict[str, Any]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    num_positions: int = 1,
    queue_max_size: int = 10000,
    num_engine_workers: int = 2,
    sleep_after_get: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> None:

    if logger is None:
        logger = logging.getLogger(__name__)

    # Build the result file path
    if result_file_path is None:
        result_file_path = Path("results") / f"{network_name}.csv"

    # Create all required queues
    engine_queue_in = asyncio.Queue(maxsize=queue_max_size)
    engine_queue_out = asyncio.Queue(maxsize=queue_max_size)

    # Create all data processing tasks
    data_generator_task = asyncio.create_task(
        create_positions(
            queues=[engine_queue_in],
            data_generator=data_generator,
            transformation_functions=transformation_functions,
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            engine_queue=engine_queue_out,
            num_transforms=len(transformation_functions) + 1,
            file_path=result_file_path,
            sleep_after_get=sleep_after_get,
            identifier_str="CANDIDATE_EVALUATION",
        )
    )

    # Create all analysis tasks
    analysis_tasks = [
        asyncio.create_task(
            analyze_position(
                consumer_queue=engine_queue_in,
                producer_queue=engine_queue_out,
                search_limits=search_limits,
                engine_generator=engine_generator,
                network_name=network_name,
                sleep_after_get=sleep_after_get,
                identifier_str=f"ANALYSIS_{index}",
            )
        )
        for index in range(num_engine_workers)
    ]

    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    for task in [
        data_generator_task,
        candidate_evaluation_task,
    ] + analysis_tasks:
        task.add_done_callback(handle_task_exception)

    # Wait for data generator task to finish
    await asyncio.wait([data_generator_task])

    # Wait for data queues to become empty
    await engine_queue_in.join()
    await engine_queue_out.join()

    # Cancel all remaining tasks
    candidate_evaluation_task.cancel()
    for analysis_task in analysis_tasks:
        analysis_task.cancel()


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
    parser.add_argument("--seed",                           type=int, default=42)  # noqa: E501
    parser.add_argument("--engine_config_name",             type=str, default="local_400_nodes.ini")  # noqa: E501
    # parser.add_argument("--engine_config_name",             type=str, default="remote_400_nodes.ini")  # noqa: E501
    parser.add_argument("--data_config_name",               type=str, default="database.ini")  # noqa: E501
    parser.add_argument("--num_positions",                  type=int, default=100_000)  # noqa: E501
    # parser.add_argument("--num_positions",                  type=int, default=100)  # noqa: E501
    # parser.add_argument("--network_path",                   type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa: E501
    parser.add_argument("--network_path",                   type=str, default="network_600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2")  # noqa: E501
    parser.add_argument("--transformations",                type=str, default=["rot90", "rot180", "rot270", "flip_diag", "flip_anti_diag", "flip_hor", "flip_vert"], nargs="+",  # noqa: E501
                                                            choices=["rot90", "rot180", "rot270", "flip_diag", "flip_anti_diag", "flip_hor", "flip_vert"])  # noqa: E501 E127
    parser.add_argument("--queue_max_size",                 type=int, default=10000)  # noqa: E501
    parser.add_argument("--num_engine_workers",             type=int, default=2)  # noqa: E501
    parser.add_argument("--result_subdir",                  type=str, default="")  # noqa: E501
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

    # Build the engine generator
    engine_config = get_engine_config(
        config_name=args.engine_config_name, config_folder_path=config_folder_path
    )
    engine_generator = get_engine_generator(engine_config)

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
    engine_config_name = args.engine_config_name[:-4]
    data_config_name = args.data_config_name[:-4]

    # Build the result file path
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}.txt"
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

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        transformation_invariance_testing(
            engine_generator=engine_generator,
            network_name=args.network_path if args.network_path else None,
            data_generator=data_generator,
            transformation_functions=transformation_functions,
            search_limits=engine_config.search_limits,
            result_file_path=result_file_path,
            num_positions=args.num_positions,
            queue_max_size=args.queue_max_size,
            num_engine_workers=args.num_engine_workers,
            sleep_after_get=0.1,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
