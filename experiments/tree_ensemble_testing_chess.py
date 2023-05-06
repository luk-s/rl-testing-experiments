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
from rl_testing.mcts.mcts_chess import (
    ChessEnsembleEvaluator,
    ChessMCTSBot,
    ChessSearchNode,
)
from rl_testing.util.chess import cp2q
from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import get_task_result_handler

RESULT_DIR = Path(__file__).parent / Path("results/tree_ensemble_testing")


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
                        info["pv"][0],
                    )
                )
        finally:
            consumer_queue.task_done()
            board_counter += 1


async def evaluate_candidates(
    reference_queue: asyncio.Queue,
    ensemble_queue: asyncio.Queue,
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
        csv_header = "fen,score_reference,move_reference,score_ensemble,move_ensemble\n"
        file.write(csv_header)

        while True:
            # Fetch the next board
            (
                board_reference,
                score_reference,
                move_reference,
            ) = await reference_queue.get()
            board_ensemble, score_ensemble, move_ensemble = await ensemble_queue.get()
            assert board_reference.fen(en_passant="fen") == board_ensemble.fen(en_passant="fen")
            fen = board_reference.fen(en_passant="fen")

            await asyncio.sleep(delay=sleep_after_get)

            logging.info(f"[{identifier_str}] Saving {board_counter}: " + fen)

            # Write the found adversarial example into a file
            result_str = (
                f"{fen},{score_reference},{move_reference},{score_ensemble},{move_ensemble}\n"
            )

            # Write the result to the file
            file.write(result_str)

            # Mark the element as processed
            reference_queue.task_done()
            ensemble_queue.task_done()

            board_counter += 1


async def ensemble_analysis(
    consumer_queue: asyncio.Queue,
    producer_queue: asyncio.Queue,
    search_limits: Dict[str, Any],
    engine_generator: EngineGenerator,
    network_names: List[str],
    logger: logging.Logger,
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    board_counter = 1

    # Create the board evaluator
    board_evaluator = ChessEnsembleEvaluator(logger=logger)

    # Initialize the board evaluator engine tasks
    num_tasks = len(network_names)
    await board_evaluator.initialize_engine_tasks(
        engine_generators=[engine_generator] * num_tasks,
        network_names=network_names,
    )

    # Create the MCTS bot
    mcts_bot = ChessMCTSBot(
        max_simulations=search_limits["nodes"],
        evaluator=board_evaluator,
        solve=True,
        child_selection_fn=ChessSearchNode.puct_value,
        dirichlet_noise=None,
        verbose=True,
        dont_return_chance_node=True,
    )

    while True:
        # Fetch the next base board, the next transformed board, and the corresponding
        # transformation index
        board = await consumer_queue.get()
        await asyncio.sleep(delay=sleep_after_get)

        logging.info(
            f"[{identifier_str}] Analyzing board {board_counter}: " + board.fen(en_passant="fen")
        )
        root_node = await mcts_bot.mcts_search(board=board)
        if root_node == "invalid":
            await producer_queue.put((board, "invalid", "invalid"))
        else:
            score = root_node.q_value()
            best_move = root_node.best_child().action
            await producer_queue.put(
                (
                    board,
                    score,
                    best_move,
                )
            )
        consumer_queue.task_done()
        board_counter += 1


async def tree_ensemble_testing(
    engine_generator_ensemble: EngineGenerator,
    engine_generator_reference: EngineGenerator,
    network_name_reference: Optional[str],
    network_names_ensemble: List[str],
    data_generator: BoardGenerator,
    search_limits_ensemble: Dict[str, Any],
    search_limits_reference: Dict[str, Any],
    result_file_path: Optional[Union[str, Path]] = None,
    num_positions: int = 1,
    queue_max_size: int = 10_000,
    sleep_after_get: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> None:

    if logger is None:
        logger = logging.getLogger(__name__)

    # Build the result file path
    if result_file_path is None:
        result_file_path = Path("results") / "results.csv"

    # Create all required queues
    board_queue_reference = asyncio.Queue(maxsize=queue_max_size)
    board_queue_ensemble = asyncio.Queue(maxsize=queue_max_size)
    result_queue_reference = asyncio.Queue(maxsize=queue_max_size)
    result_queue_ensemble = asyncio.Queue(maxsize=queue_max_size)

    # Create all data processing tasks
    data_generator_task = asyncio.create_task(
        create_positions(
            queues=[board_queue_reference, board_queue_ensemble],
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            reference_queue=result_queue_reference,
            ensemble_queue=result_queue_ensemble,
            file_path=result_file_path,
            sleep_after_get=sleep_after_get,
            identifier_str="CANDIDATE_EVALUATION",
        )
    )

    analysis_task = asyncio.create_task(
        analyze_position(
            consumer_queue=board_queue_reference,
            producer_queue=result_queue_reference,
            search_limits=search_limits_reference,
            engine_generator=engine_generator_reference,
            network_name=network_name_reference,
            sleep_after_get=sleep_after_get,
            identifier_str="CLASSIC_ANALYSIS",
        )
    )

    ensemble_task = asyncio.create_task(
        ensemble_analysis(
            consumer_queue=board_queue_ensemble,
            producer_queue=result_queue_ensemble,
            search_limits=search_limits_ensemble,
            engine_generator=engine_generator_ensemble,
            network_names=network_names_ensemble,
            logger=logger,
            sleep_after_get=sleep_after_get,
            identifier_str="ENSEMBLE_ANALYSIS",
        )
    )

    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    for task in [
        data_generator_task,
        candidate_evaluation_task,
        analysis_task,
        ensemble_task,
    ]:
        task.add_done_callback(handle_task_exception)

    # Wait for data generator task to finish
    await asyncio.wait([data_generator_task])

    # Wait for data queues to become empty
    await board_queue_reference.join()
    await board_queue_ensemble.join()
    await result_queue_reference.join()
    await result_queue_ensemble.join()

    # Cancel all remaining tasks
    candidate_evaluation_task.cancel()
    analysis_task.cancel()
    ensemble_task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    network_paths = [
        "T608927-c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be",
        "T611246-7ca2381cfeac5c280f304e7027ffbea1b7d87474672e5d6fb16d5cd881640e04",
        "T771717-d8ae0251e0924eb2b5d2853c6aad7926f4b676ea59ebb3ea2b2cc469eac9bdda",
        "T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2",
        "T807301-c85375d37b369db8db6b0665d12647e7a7a3c9453f5ba46235966bc2ed433638",
    ]

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
    parser.add_argument("--engine_config_name_ensemble",    type=str, default="local_100_nodes_debug.ini")  # noqa: E501
    parser.add_argument("--engine_config_name_reference",   type=str, default="local_500_nodes_debug.ini")  # noqa: E501
    # parser.add_argument("--engine_config_name_ensemble",    type=str, default="remote_debug_100_nodes.ini")  # noqa: E501
    # parser.add_argument("--engine_config_name_reference",   type=str, default="remote_debug_500_nodes.ini")  # noqa: E501
    parser.add_argument("--data_config_name",               type=str, default="database.ini")  # noqa: E501
    # parser.add_argument("--data_config_name",               type=str, default="database_2000_games_read.ini")  # noqa: E501
    parser.add_argument("--num_positions",                  type=int, default=100_000)  # noqa: E501
    # parser.add_argument("--num_positions",                  type=int, default=100)  # noqa: E501
    # parser.add_argument("--network_path",                   type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa: E501
    parser.add_argument("--network_path_reference",         type=str, default='T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2')  # noqa: E501
    parser.add_argument("--network_paths_ensemble",         type=str, default=network_paths, nargs="+", choices=network_paths)  # noqa: E501 E127
    parser.add_argument("--queue_max_size",                 type=int, default=10000)  # noqa: E501
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

    # Build the ensemble engine generator
    engine_config_ensemble = get_engine_config(
        config_name=args.engine_config_name_ensemble,
        config_folder_path=config_folder_path,
    )
    engine_generator_ensemble = get_engine_generator(engine_config_ensemble)

    # Build the reference engine generator
    engine_config_reference = get_engine_config(
        config_name=args.engine_config_name_reference,
        config_folder_path=config_folder_path,
    )
    engine_generator_reference = get_engine_generator(engine_config_reference)

    # Build the data generator
    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Create results-file-name
    engine_config_name_ensemble = args.engine_config_name_ensemble[:-4]
    engine_config_name_reference = args.engine_config_name_reference[:-4]
    data_config_name = args.data_config_name[:-4]
    num_games_read = data_config.games_read
    num_network_names_ensemble = len(args.network_paths_ensemble)

    # Build the result file path
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_REFERENCE_ENGINE_{engine_config_name_reference}_ENSEMBLE_ENGINE_"
        f"{engine_config_name_ensemble}_DATA_{data_config_name}_"
        f"ENSEMBLE_SIZE_{num_network_names_ensemble}_NUM_GAMES_READ_{num_games_read}.txt"
    )

    # Store the experiment configuration in the result file
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        tree_ensemble_testing(
            engine_generator_ensemble=engine_generator_ensemble,
            engine_generator_reference=engine_generator_reference,
            network_name_reference=args.network_path_reference
            if args.network_path_reference
            else None,
            network_names_ensemble=args.network_paths_ensemble,
            data_generator=data_generator,
            search_limits_ensemble=engine_config_ensemble.search_limits,
            search_limits_reference=engine_config_reference.search_limits,
            result_file_path=result_file_path,
            num_positions=args.num_positions,
            queue_max_size=args.queue_max_size,
            sleep_after_get=0.1,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
