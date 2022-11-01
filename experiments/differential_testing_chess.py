import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import numpy as np
from rl_testing.config_parsers import BoardGeneratorConfig, RemoteEngineConfig
from rl_testing.config_parsers.engine_config_parser import EngineConfig
from rl_testing.data_generators import (
    BoardGenerator,
    DataBaseBoardGenerator,
    FENDatabaseBoardGenerator,
    RandomBoardGenerator,
)
from rl_testing.engine_generators import EngineGenerator, RemoteEngineGenerator
from rl_testing.engine_generators.relaxed_uci_protocol import RelaxedUciProtocol
from rl_testing.util.util import MoveStat, PositionStat, parse_info, plot_board

RESULT_DIR = Path(__file__).parent / Path("results/differential_testing")


async def analyze_positions(
    queue: asyncio.Queue,
    engine: RelaxedUciProtocol,
    search_limits: Dict[str, Any],
    num_boards: int,
    sleep_after_get: float = 0.0,
    engine_generator: Optional[EngineGenerator] = None,
    network_name: Optional[str] = None,
) -> List[Tuple[Union[chess.Move, str], Dict[chess.Move, MoveStat], List[PositionStat]]]:
    results = []
    # Iterate over all boards
    for board_index in range(num_boards):
        # Fetch the next board from the queue
        board = await queue.get()

        await asyncio.sleep(delay=sleep_after_get)

        print(f"Analyzing board {board_index + 1}/{num_boards}: " + board.fen(en_passant="fen"))

        move_stats, position_stats = {}, []

        # Needs to be in a try-except because the engine might crash unexpectedly
        try:
            # Start the analysis
            with await engine.analysis(
                board, limit=chess.engine.Limit(**search_limits)
            ) as analysis:
                async for info in analysis:
                    # Parse the analysis info
                    result = parse_info(info=info, raise_exception=False)

                    # Check if the info could be parsed into a 'MoveStat'
                    # or a 'PositionStat' instance
                    if result is not None:
                        if isinstance(result, MoveStat):
                            move_stats[result.move] = result
                        elif isinstance(result, PositionStat):
                            position_stats.append(result)
                        else:
                            ValueError(f"Objects of type {type(result)} are not supported")

        except chess.engine.EngineTerminatedError:
            if engine_generator is None or network_name is None:
                print("Can't restart engine due to missing generator")
                raise

            # Mark the current board as failed
            results.append(("invalid", {}, []))

            # Try to restart the engine
            print("Trying to restart engine")

            engine_generator.set_network(network_name)
            engine = await engine_generator.get_initialized_engine()

        else:
            # Check if the proposed best move is valid
            if engine.invalid_best_move:
                best_move = (await analysis.wait()).move
                results.append(("invalid", {}, []))
            else:
                best_move = (await analysis.wait()).move
                results.append((best_move, move_stats, position_stats))
        finally:
            queue.task_done()

    return results


async def get_positions_async(
    queues: List[asyncio.Queue],
    data_generator: BoardGenerator,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
) -> List[chess.Board]:
    boards = []

    # Create random chess positions if necessary
    for board_index in range(num_positions):

        # Create a random chess position
        board_candidate = data_generator.next()

        # Check if the generated position was valid
        if board_candidate != "failed":
            boards.append(board_candidate)
            fen = board_candidate.fen(en_passant="fen")
            print(f"Created board {board_index}: " f"{fen}")
            for queue in queues:
                await queue.put(board_candidate.copy())

            await asyncio.sleep(delay=sleep_between_positions)

    return boards


async def differential_testing(
    network_name1: str,
    network_name2: str,
    engine_generator: EngineGenerator,
    data_generator: BoardGenerator,
    *,
    search_limits: Optional[Dict[str, Any]] = None,
    num_positions: int = 1,
    sleep_between_positions: float = 0.2,
) -> Tuple[List[chess.Board], List[Tuple[float, float]]]:

    engine_generator.set_network(network_name1)
    engine1 = await engine_generator.get_initialized_engine()

    engine_generator.set_network(network_name2)
    engine2 = await engine_generator.get_initialized_engine()

    # If no search limit has been specified, just search one single node
    if search_limits is None:
        search_limits = {"nodes": 1}

    results = []
    boards_final = []
    queue1, queue2 = asyncio.Queue(), asyncio.Queue()

    data_generator_task = asyncio.create_task(
        get_positions_async(
            queues=[queue1, queue2],
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_between_positions,
        )
    )

    (data_consumer_task1, data_consumer_task2) = [
        asyncio.create_task(
            analyze_positions(
                queue=queue,
                engine=engine,
                search_limits=search_limits,
                num_boards=num_positions,
                sleep_after_get=sleep_time,
                engine_generator=engine_generator,
                network_name=network_name,
            )
        )
        for queue, engine, sleep_time, network_name in zip(
            [queue1, queue2], [engine1, engine2], [0.0, 0.0], [network_name1, network_name2]
        )
    ]

    # Wait until all tasks finish
    boards, results1, results2 = await asyncio.gather(
        data_generator_task, data_consumer_task1, data_consumer_task2
    )

    # await engine1.quit()
    # await engine2.quit()

    for board_index, board in enumerate(boards):
        best_move1, move_stats1, _ = results1[board_index]
        best_move2, move_stats2, _ = results2[board_index]

        if best_move1 != "invalid" and best_move2 != "invalid":
            results.append((move_stats1[best_move1].Q, move_stats2[best_move2].Q))
            boards_final.append(board)
        else:
            message = "board: " + board.fen(en_passant="fen") + " "
            if best_move1 == "invalid":
                message += "engine 1 invalid, "
            if best_move2 == "invalid":
                message += "engine 2 invalid"
            print(message)

    return boards_final, results


def analyze_results(
    boards: List[chess.Board],
    results: List[Tuple[float, float]],
    difference_threshold: float = 1.0,
    show: bool = True,
    result_dir: Union[str, Path] = "",
    fontsize: int = 22,
) -> Tuple[chess.Board, float, float, float]:
    # Find results where the models have a large difference
    results = np.array(results)
    differences = np.abs(results[:, 0] - results[:, 1])
    indices = np.where(differences >= difference_threshold)[0]

    special_results = []

    for index in indices:
        board = boards[index]
        value1, value2 = results[index]
        special_results.append((board, differences[index], value1, value2))
        title = f"value1: {value1}, value2: {value2}"
        fen = board.fen(en_passant="fen")
        fen = fen.replace(" ", "_")
        fen = fen.replace("/", "|")
        file_path = Path(result_dir) / Path(fen + ".png")
        plot_board(board, title=title, fen=fen, show=show, save_path=file_path, fontsize=fontsize)

    return special_results


if __name__ == "__main__":
    ##################################
    #           CONFIG START         #
    ##################################
    SEED = 42
    ENGINE_CONFIG_NAME = "local_400_nodes.ini"  # "remote_400_nodes.ini"
    DATA_CONFIG_NAME = "random_fen_database_test.ini"  # "random_many_pieces.ini"
    REMOTE = False
    POSITIONS = []
    NUM_POSITIONS = 10
    # DIFFERENCE_THRESHOLD = 1
    NETWORK_PATH1 = "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    NETWORK_PATH2 = "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    # NETWORK_PATH1 = "f21ee51844a7548c004a1689eacd8b4cd4c6150d6e03c732b211cf9963d076e1"
    # NETWORK_PATH2 = "fbd5e1c049d5a46c098f0f7f12e79e3fb82a7a6cd1c9d1d0894d0aae2865826f"

    RESULT_SUBDIR = "main_experiment"
    ##################################
    #           CONFIG END           #
    ##################################

    np.random.seed(SEED)

    # Build the engine generator
    if REMOTE:
        RemoteEngineConfig.set_config_folder_path(
            Path(__file__).parent.absolute() / Path("configs/engine_configs/")
        )
        engine_config = RemoteEngineConfig.from_config_file(ENGINE_CONFIG_NAME)
        engine_generator = RemoteEngineGenerator(engine_config)

    else:
        EngineConfig.set_config_folder_path(
            Path(__file__).parent.absolute() / Path("configs/engine_configs/")
        )
        engine_config = EngineConfig.from_config_file(ENGINE_CONFIG_NAME)
        engine_generator = EngineGenerator(engine_config)

    # Build the data generator
    BoardGeneratorConfig.set_config_folder_path(
        Path(__file__).parent.absolute() / Path("configs/data_generator_configs")
    )
    data_config = BoardGeneratorConfig.from_config_file(DATA_CONFIG_NAME)
    # data_generator = RandomBoardGenerator(**data_config.board_generator_config)
    data_generator = FENDatabaseBoardGenerator(**data_config.board_generator_config)

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    boards, results = asyncio.run(
        differential_testing(
            network_name1=NETWORK_PATH1,
            network_name2=NETWORK_PATH2,
            data_generator=data_generator,
            engine_generator=engine_generator,
            search_limits=engine_config.search_limits,
            num_positions=NUM_POSITIONS,
        )
    )

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time: .3f} seconds")

    # Create results-file-name
    engine_config_name = ENGINE_CONFIG_NAME[:-4]
    data_config_name = DATA_CONFIG_NAME[:-4]

    # Open the results file
    result_directory = RESULT_DIR / RESULT_SUBDIR
    result_directory.mkdir(parents=True, exist_ok=True)
    with open(
        result_directory
        / Path(f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}.txt"),
        "a",
    ) as f:
        # Store the config
        f.write(f"{SEED = }\n")
        f.write(f"{ENGINE_CONFIG_NAME = }\n")
        f.write(f"{DATA_CONFIG_NAME = }\n")
        f.write(f"{REMOTE = }\n")
        f.write(f"{POSITIONS = }\n")
        f.write(f"{NUM_POSITIONS = }\n")
        f.write(f"{NETWORK_PATH1 = }\n")
        f.write(f"{NETWORK_PATH2 = }\n")
        f.write(f"{RESULT_SUBDIR = }\n")
        f.write("\n")

        # Store the results
        f.write("FEN, Q1, Q2\n")
        for board, (q1, q2) in zip(boards, results):
            fen = board.fen(en_passant="fen").replace(" ", "_")
            f.write(f"{fen},{q1},{q2}\n")

    """
    if results:
        special_cases = analyze_results(
            boards,
            results,
            difference_threshold=DIFFERENCE_THRESHOLD,
            show=False,
            result_dir=RESULT_DIR / RESULT_SUBDIR,
            fontsize=14,
        )
        print(
            f"Found {len(special_cases)} boards where "
            f"evaluation differs by more than {DIFFERENCE_THRESHOLD}!"
        )
        for board, difference, v1, v2 in special_cases:
            fen = board.fen(en_passant="fen")
            print(f"{fen}, {difference = :.2f}, {v1 = :.3f}, {v2 = :.3f}")
    """
