import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.mcts.tree_parser import TreeInfo
from rl_testing.util.chess import cp2q
from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import get_task_result_handler

RESULT_DIR = Path(__file__).parent / Path("results/tree_stat_testing")


async def create_positions(
    queues: List[asyncio.Queue],
    data_generator: BoardGenerator,
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

        # Check if the generated position has at least two legal moves and if this position
        # has not been generated before
        if board_candidate.fen() not in fen_cache and len(legal_moves) > 1:
            fen_cache[board_candidate.fen()] = True

            # Log the base position
            fen = board_candidate.fen(en_passant="fen")
            logging.info(f"[{identifier_str}] Created base board {board_index + 1}: {fen}")

            # Send the base position to all queues
            for queue in queues:
                await queue.put(board_candidate.copy())

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
        board = await consumer_queue.get()
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
            assert "mcts_tree" in info
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
            await producer_queue.put((board, "invalid", "invalid"))
        else:
            score_cp = info["score"].relative.score(mate_score=12780)

            # Check if the computed score is valid
            if engine_generator is not None and not engine_generator.cp_score_valid(score_cp):
                await producer_queue.put((board, "invalid", "invalid"))

            else:
                # Add the board to the receiver queue
                # The 12780 is used as maximum value because we use the q2cp function
                # to convert q_values to centipawns. This formula has values in
                # [-12780, 12780] for q_values in [-1, 1]
                await producer_queue.put(
                    (
                        board,
                        cp2q(score_cp),
                        info["mcts_tree"],
                    )
                )
        finally:
            consumer_queue.task_done()
            board_counter += 1


async def evaluate_candidates(
    result_queue: asyncio.Queue,
    file_path: Union[str, Path],
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    # Create a file to store the results
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    board_counter = 1

    with open(file_path, "a") as file:

        # Store the header of the result data in the result file
        csv_header = "fen;score;edge_info\n"
        file.write(csv_header)  # noqa: E501

        while True:
            # Fetch the next board and the corresponding scores from the queues
            board, score, tree = await result_queue.get()
            fen = board.fen(en_passant="fen")

            await asyncio.sleep(delay=sleep_after_get)

            logging.info(f"[{identifier_str}] Saving board {board_counter}: " + fen)

            # Write the found adversarial example into a file
            result_str = f"{fen};{score};"

            # Extract information about all outgoing edges of the root node
            if isinstance(tree, TreeInfo):
                # Extract the information about the root node
                root_node = tree.root_node
                assert root_node is not None

                # Extract the edges and sort them by their move
                edges = [e for e in root_node.child_edges if e.end_node is not None]
                moves = [edge.move.uci() for edge in edges]
                edge_tuples = sorted(zip(moves, edges), key=lambda x: x[0])

                # Store information about the edges
                for move, edge in edge_tuples:
                    result_str += ",".join(
                        [
                            move,
                            str(edge.num_visits),
                            str(edge.in_flight_visits),
                            str(edge.policy_value),
                            str(edge.v_value),
                            str(edge.q_value),
                            str(edge.u_value),
                            str(edge.s_value),
                        ]
                    )
            else:
                result_str += "invalid"

            # Mark the element as processed
            result_queue.task_done()

            result_str += "\n"

            # Write the result to the file
            file.write(result_str)

            board_counter += 1


async def tree_stat_testing(
    engine_generator: EngineGenerator,
    network_name: Optional[str],
    data_generator: BoardGenerator,
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
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            result_queue=engine_queue_out,
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
    # parser.add_argument("--engine_config_name",             type=str, default="remote_debug_400_nodes.ini")  # noqa: E501
    parser.add_argument("--data_config_name",               type=str, default="database.ini")  # noqa: E501
    parser.add_argument("--num_positions",                  type=int, default=100_000)  # noqa: E501
    # parser.add_argument("--num_positions",                  type=int, default=100)  # noqa: E501
    # parser.add_argument("--network_path",                   type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa: E501
    parser.add_argument("--network_path",                   type=str, default="")  # noqa: E501
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

    # Build the data generator
    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Create results-file-name
    engine_config_name = args.engine_config_name[:-4]
    data_config_name = args.data_config_name[:-4]
    network_name = args.network_path[:7]

    # Build the result file path
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}_NET_{network_name}.txt"
    )

    # Store the experiment configuration in the result file
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        tree_stat_testing(
            engine_generator=engine_generator,
            network_name=args.network_path if args.network_path else None,
            data_generator=data_generator,
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
