import argparse
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.engine_generators.relaxed_uci_protocol import RelaxedUciProtocol
from rl_testing.util.util import (
    MoveStat,
    PositionStat,
    get_task_result_handler,
    cp2q,
)
from rl_testing.util.experiment import store_experiment_params

RESULT_DIR = Path(__file__).parent / Path("results/differential_testing")


async def get_positions(
    queues: List[asyncio.Queue],
    data_generator: BoardGenerator,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    identifier_str: str = "",
) -> None:
    board_cache = {}

    # Create random chess positions if necessary
    board_index = 0
    while board_index < num_positions:

        # Create a random chess position
        board_candidate = data_generator.next()

        # Check if the generated position was valid
        if board_candidate != "failed":
            fen = board_candidate.fen(en_passant="fen")
            if fen in board_cache:
                continue
            board_cache[fen] = True
            logging.info(f"[{identifier_str}] Created board {board_index}: " f"{fen}")
            for queue in queues:
                await queue.put(board_candidate.copy())

            await asyncio.sleep(delay=sleep_between_positions)
            board_index += 1


async def analyze_positions(
    input_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
    engine: RelaxedUciProtocol,
    search_limits: Dict[str, Any],
    sleep_after_get: float = 0.0,
    engine_generator: Optional[EngineGenerator] = None,
    network_name: Optional[str] = None,
    identifier_str: str = "",
) -> List[
    Tuple[Union[chess.Move, str], Dict[chess.Move, MoveStat], List[PositionStat]]
]:
    board_index = 0
    # Required to ensure that the engine doesn't use cached results from
    # previous analyses
    analysis_counter = 0

    while True:
        # Fetch the next board from the queue
        board = await input_queue.get()
        fen = board.fen(en_passant="fen")
        await asyncio.sleep(delay=sleep_after_get)

        logging.info(f"[{identifier_str}] Analyzing board {board_index + 1}: {fen}")

        # Needs to be in a try-except because the engine might crash unexpectedly
        try:
            analysis_counter += 1
            info = await engine.analyse(
                board, chess.engine.Limit(**search_limits), game=analysis_counter
            )
            best_move = info["pv"][0]
            score = cp2q(info["score"].relative.score(mate_score=12800))

        except chess.engine.EngineTerminatedError:
            if engine_generator is None or network_name is None:
                logging.info(
                    f"[{identifier_str}]Can't restart engine due to missing generator"
                )
                raise

            # Mark the current board as failed
            await output_queue.put((fen, "invalid", "invalid"))

            # Try to restart the engine
            logging.info(f"[{identifier_str}] Trying to restart engine")

            engine_generator.set_network(network_name)
            engine = await engine_generator.get_initialized_engine()

        else:
            # Check if the proposed best move is valid
            if engine.invalid_best_move:
                await output_queue.put((fen, "invalid", "invalid"))
            else:
                await output_queue.put((fen, best_move, score))
        finally:
            input_queue.task_done()

        board_index += 1


async def analyze_results(
    input_queues: List[asyncio.Queue],
    file_path: Union[str, Path],
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    num_queues = len(input_queues)

    # Create a file to store the results
    with open(file_path, "a") as file:
        file_header = ["fen"] + [
            f"best_move{i},score{i}" for i in range(1, num_queues + 1)
        ]
        file.write(",".join(file_header) + "\n")

        # Repeatedly fetch results from the queues
        while True:
            fens, best_moves, scores = [], [], []
            for queue in input_queues:
                fen, best_move, score = await queue.get()
                fens.append(fen)
                best_moves.append(best_move)
                scores.append(score)
                await asyncio.sleep(delay=sleep_after_get)

            # assert that all elements of the fen list are equal
            assert all(fen == fens[0] for fen in fens)

            logging.info(f"[{identifier_str}] Received results for {fens[0]}")

            # Write the results to the file
            result_list = [fens[0]] + [
                f"{best_moves[i]},{scores[i]}" for i in range(num_queues)
            ]
            file.write(",".join(result_list) + "\n")


async def differential_testing(
    network_name1: str,
    network_name2: str,
    engine_generator: EngineGenerator,
    data_generator: BoardGenerator,
    result_file_path: Union[str, Path],
    *,
    search_limits: Optional[Dict[str, Any]] = None,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> None:

    # Set up the logger if necessary
    if logger is None:
        logger = logging.getLogger(__name__)

    engine_generator.set_network(network_name1)
    engine1 = await engine_generator.get_initialized_engine()

    engine_generator.set_network(network_name2)
    engine2 = await engine_generator.get_initialized_engine()

    # If no search limit has been specified, just search one single node
    if search_limits is None:
        search_limits = {"nodes": 1}

    # Create the queues
    board_queue1, board_queue2 = asyncio.Queue(), asyncio.Queue()
    result_queue1, result_queue2 = asyncio.Queue(), asyncio.Queue()

    # Create the board generator task
    data_generator_task = asyncio.create_task(
        get_positions(
            queues=[board_queue1, board_queue2],
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_between_positions,
            identifier_str="BOARD GENERATOR",
        )
    )

    # Create the board analysis tasks
    (data_consumer_task1, data_consumer_task2) = [
        asyncio.create_task(
            analyze_positions(
                input_queue=input_queue,
                output_queue=output_queue,
                engine=engine,
                search_limits=search_limits,
                sleep_after_get=sleep_time,
                engine_generator=engine_generator,
                network_name=network_name,
                identifier_str=identifier_str,
            )
        )
        for input_queue, output_queue, engine, sleep_time, network_name, identifier_str in zip(
            [board_queue1, board_queue2],
            [result_queue1, result_queue2],
            [engine1, engine2],
            [0.05, 0.05],
            [network_name1, network_name2],
            ["ANALYSIS 1", "ANALYSIS 2"],
        )
    ]

    # Create the result analysis task
    analysis_task = asyncio.create_task(
        analyze_results(
            input_queues=[result_queue1, result_queue2],
            file_path=result_file_path,
            sleep_after_get=0.1,
            identifier_str="RESULT",
        )
    )

    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    for task in [
        data_generator_task,
        data_consumer_task1,
        data_consumer_task2,
        analysis_task,
    ]:
        task.add_done_callback(handle_task_exception)

    # Wait for the data generator to finish
    await asyncio.wait([data_generator_task])

    # Wait for the queues to be empty
    await board_queue1.join()
    await board_queue2.join()
    await result_queue1.join()
    await result_queue2.join()

    # Cancel all tasks
    for task in [
        data_generator_task,
        data_consumer_task1,
        data_consumer_task2,
        analysis_task,
    ]:
        task.cancel()


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
    parser.add_argument("--seed",               type=int, default=42)
    # parser.add_argument("--engine_config_name", type=str, default="remote_400_nodes.ini")
    parser.add_argument("--engine_config_name", type=str, default="local_400_nodes.ini")
    parser.add_argument("--data_config_name",   type=str, default="database.ini")
    parser.add_argument("--num_positions",      type=int, default=100_000)
    parser.add_argument("--network_path1",      type=str, default="T807301-c85375d37b369db8db6b0665d12647e7a7a3c9453f5ba46235966bc2ed433638")  # noqa: E501
    # parser.add_argument("--network_path2",      type=str, default="network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be")  # noqa: E501
    parser.add_argument("--network_path2",      type=str, default="T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2")  # noqa: E501
    parser.add_argument("--result_subdir",      type=str, default="main_results")
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

    # Parse the arguments
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Get the engine config and engine generator
    engine_config = get_engine_config(
        config_name=args.engine_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/engine_configs/"),
    )
    engine_generator = get_engine_generator(engine_config)

    # Get the data config and data generator
    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Create results-file-name
    engine_config_name = args.engine_config_name[:-4]
    data_config_name = args.data_config_name[:-4]

    # Store the experiment config in the results file
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}.txt"
    )
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        differential_testing(
            network_name1=args.network_path1,
            network_name2=args.network_path2,
            data_generator=data_generator,
            engine_generator=engine_generator,
            search_limits=engine_config.search_limits,
            num_positions=args.num_positions,
            result_file_path=result_file_path,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
