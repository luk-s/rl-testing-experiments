import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import cp2q, get_task_result_handler

RESULT_DIR = Path(__file__).parent / Path("results/parent_child_testing")


class ReceiverCache:
    def __init__(self, queue: asyncio.Queue) -> None:
        assert isinstance(queue, asyncio.Queue)
        self.queue = queue

        self.score_cache = {}

    async def receive_data(self) -> List[Iterable[Any]]:
        # Receive data from queue
        base_board, num_child_nodes, board, move, score = await self.queue.get()

        assert isinstance(base_board, chess.Board)

        # The boards might not arrive in the correct order due to the asynchronous nature of
        # the program. Therefore, we need to cache the boards and scores until we have all
        # of them.
        fen = base_board.fen()
        move_str = move.uci() if move is not None else "NaN"
        if fen in self.score_cache:
            self.score_cache[fen].append((move_str, score))
        else:
            self.score_cache[fen] = [(move_str, score)]

        complete_data_tuples = []
        # Check if we have all the data for this board
        if len(self.score_cache[fen]) == num_child_nodes:
            # We have all the data for this board
            complete_data_tuples.append((fen, self.score_cache[fen]))
            del self.score_cache[fen]

        return complete_data_tuples


async def create_positions(
    queues: List[asyncio.Queue],
    data_generator: BoardGenerator,
    max_child_moves: int = 10,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    identifier_str: str = "",
) -> None:
    fen_cache = {}

    board_index = 1
    while board_index <= num_positions:

        # Create a random chess position
        board_candidate = data_generator.next()

        # Get the number of legal moves for the current board
        if board_candidate == "failed":
            continue

        legal_moves = list(board_candidate.legal_moves)

        # Check if the generated position should be further processed
        if board_candidate.fen() not in fen_cache and 0 < len(legal_moves) <= max_child_moves:
            fen_cache[board_candidate.fen()] = True

            # Get all child positions which are not terminal positions
            board_list = [board_candidate]
            move_list = [None]
            for legal_move in legal_moves:
                board_temp = board_candidate.copy()
                board_temp.push(legal_move)
                if len(list(board_temp.legal_moves)) > 0:
                    board_list.append(board_temp)
                    move_list.append(legal_move)

            # If all child positions are terminal positions, we do not need to process this
            if len(board_list) == 1:
                continue

            # Log the base position
            fen = board_candidate.fen(en_passant="fen")
            logging.info(f"[{identifier_str}] Created base board {board_index + 1}: " f"{fen}")

            # Log the child positions
            for board in board_list[1:]:
                fen = board.fen(en_passant="fen")
                logging.info(f"[{identifier_str}] Created transformed board: " f"{fen}")

            # Send the base position to all queues
            for queue in queues:
                for board, move in zip(board_list, move_list):
                    await queue.put((board_candidate.copy(), len(board_list), board.copy(), move))

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
        base_board, num_child_nodes, board, move = await consumer_queue.get()
        await asyncio.sleep(delay=sleep_after_get)

        logging.info(
            f"[{identifier_str}] Analyzing board {board_counter}: " + board.fen(en_passant="fen")
        )
        try:
            # Analyze the board
            analysis_counter += 1
            info = await engine.analyse(
                board, chess.engine.Limit(**search_limits), game=analysis_counter
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
            await producer_queue.put((base_board, num_child_nodes, board, move, "invalid"))
        else:
            score_cp = info["score"].relative.score(mate_score=12800)

            # Check if the computed score is valid
            if engine_generator is not None and not engine_generator.cp_score_valid(score_cp):
                await producer_queue.put((base_board, num_child_nodes, board, move, "invalid"))

            else:
                # Add the board to the receiver queue
                # The 12800 is used as maximum value because we use the q2cp function
                # to convert q_values to centipawns. This formula has values in
                # [-12800, 12800] for q_values in [-1, 1]
                await producer_queue.put(
                    (
                        base_board,
                        num_child_nodes,
                        board,
                        move,
                        cp2q(score_cp),
                    )
                )
        finally:
            consumer_queue.task_done()
            board_counter += 1


async def evaluate_candidates(
    engine_queue: asyncio.Queue,
    max_child_moves: int,
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
    receiver_cache = ReceiverCache(queue=engine_queue)

    with open(file_path, "a") as file:
        # Create the header of the result file
        csv_header = "fen,original," + ",".join(
            [f"move_{i},score_{i}," for i in range(max_child_moves + 1)]
        )
        file.write(csv_header[:-1] + "\n")  # noqa: E501

        while True:
            # Fetch the next board and the corresponding scores from the queues
            complete_data_tuples = await receiver_cache.receive_data()

            # Iterate over the received data
            for fen, score_tuple_list in complete_data_tuples:
                await asyncio.sleep(delay=sleep_after_get)

                logging.info(f"[{identifier_str}] Saving board {board_counter}: " + fen)

                # Write the found adversarial example into a file
                index = 0
                result_str = f"{fen},"
                for (move, score) in score_tuple_list:
                    # Add the score to the result string
                    result_str += f"{move},{score},"

                    # Mark the element as processed
                    engine_queue.task_done()

                    index += 1

                # Fill the row with empty values if the number of child nodes is smaller than
                # the maximum number of child nodes
                for _ in range(index, max_child_moves):
                    result_str += "NaN,NaN,"

                result_str = result_str[:-1] + "\n"

                # Write the result to the file
                file.write(result_str)

                board_counter += 1


async def parent_child_invariance_testing(
    engine_generator: EngineGenerator,
    network_name: Optional[str],
    data_generator: BoardGenerator,
    *,
    max_child_moves: int = 10,
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
            max_child_moves=max_child_moves,
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            engine_queue=engine_queue_out,
            max_child_moves=max_child_moves,
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
    # parser.add_argument("--engine_config_name",             type=str, default="local_400_nodes.ini")  # noqa: E501
    parser.add_argument("--engine_config_name",             type=str, default="remote_400_nodes.ini")  # noqa: E501
    parser.add_argument("--data_config_name",               type=str, default="database.ini")  # noqa: E501
    parser.add_argument("--num_positions",                  type=int, default=100_000)  # noqa: E501
    # parser.add_argument("--num_positions",                  type=int, default=100)  # noqa: E501
    # parser.add_argument("--network_path",                   type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa: E501
    parser.add_argument("--network_path",                   type=str, default="network_600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2")  # noqa: E501
    parser.add_argument("--max_child_moves",                type=int, default=10),  # noqa: E501
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

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        parent_child_invariance_testing(
            engine_generator=engine_generator,
            network_name=args.network_path if args.network_path else None,
            data_generator=data_generator,
            max_child_moves=args.max_child_moves,
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
