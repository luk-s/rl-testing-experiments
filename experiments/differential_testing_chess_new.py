import asyncio
import time
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncssh
import chess
import chess.engine
import numpy as np
from rl_testing.config_parsers import BoardGeneratorConfig, RemoteEngineConfig
from rl_testing.data_generators import (
    BoardGenerator,
    DataBaseBoardGenerator,
    RandomBoardGenerator,
)
from rl_testing.util.relaxed_uci_protocol import RelaxedUciProtocol
from rl_testing.util.util import MoveStat, PositionStat, parse_info, plot_board


async def analyze_positions(
    queue: asyncio.Queue,
    engine: RelaxedUciProtocol,
    search_limits: Dict[str, Any],
    num_boards: int,
    sleep_after_get: float = 0,
) -> List[Tuple[Union[chess.Move, str], Dict[chess.Move, MoveStat], List[PositionStat]]]:
    results = []
    # Iterate over all boards
    for board_index in range(num_boards):
        # Fetch the next board from the queue
        board = await queue.get()

        await asyncio.sleep(delay=sleep_after_get)

        print(f"Analyzing board {board_index}/{num_boards}: " + board.fen(en_passant="fen"))

        move_stats, position_stats = {}, []

        # Start the analysis
        with await engine.analysis(board, limit=chess.engine.Limit(**search_limits)) as analysis:
            async for info in analysis:
                # Parse the analysis info
                result = parse_info(info=info, raise_exception=False)

                # Check if the info could be parsed into a 'MoveStat' or a 'PositionStat' instance
                if result is not None:
                    if isinstance(result, MoveStat):
                        move_stats[result.move] = result
                    elif isinstance(result, PositionStat):
                        position_stats.append(result)
                    else:
                        ValueError(f"Objects of type {type(result)} are not supported")

        # Check if the proposed best move is valid
        if engine.invalid_best_move:
            results.append(("invalid", {}, []))
        else:
            best_move = (await analysis.wait()).move
            results.append((best_move, move_stats, position_stats))

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
    remote_host: str,
    remote_user: str,
    password_required: bool,
    engine_path: str,
    engine_config: str,
    network_path1: str,
    network_path2: str,
    data_generator: BoardGenerator,
    *,
    search_limits: Optional[Dict[str, Any]] = None,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
) -> Tuple[List[chess.Board], List[Tuple[float, float]]]:
    remote_password = None

    if remote_password is None and password_required:
        remote_password = getpass(
            prompt=f"Please specify the SSH password for the user {remote_user}:\n"
        )

    # Start connection
    async with asyncssh.connect(
        remote_host, username=remote_user, password=remote_password
    ) as conn:
        del remote_password

        # Set the CUDA available devices
        # _ = await conn.run("export CUDA_VISIBLE_DEVICES=3,4", check=True)

        # Connect the two networks to test
        _, engine1 = await conn.create_subprocess(
            RelaxedUciProtocol,
            # engine_path,
            "/home/flurilu/Software/leelachesszero/lc0/build/release/lc0",
            env={"CUDA_VISIBLE_DEVICES": "3"},
        )

        _, engine2 = await conn.create_subprocess(
            RelaxedUciProtocol,
            # engine_path,
            "/home/flurilu/Software/leelachesszero/lc0/build/release/lc0",
            env={"CUDA_VISIBLE_DEVICES": "3"},
        )

        # Initialize the two engines
        await engine1.initialize()
        await engine2.initialize()

        # Configur engine 1
        engine1_config = dict(engine_config)
        engine1_config["WeightsFile"] = network_path1
        await engine1.configure(engine1_config)

        # Configure engine 2
        engine2_config = dict(engine_config)
        engine2_config["WeightsFile"] = network_path2
        await engine2.configure(engine2_config)

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
        """
        data_consumer_task1 = asyncio.create_task(
            analyze_positions(
                queue=queue1,
                engine=engine1,
                search_limits=search_limits,
                num_boards=num_positions,
                sleep_after_get=0,
            )
        )
        """
        (data_consumer_task1, data_consumer_task2) = [
            asyncio.create_task(
                analyze_positions(
                    queue=queue,
                    engine=engine,
                    search_limits=search_limits,
                    num_boards=num_positions,
                    sleep_after_get=sleep_time,
                )
            )
            for queue, engine, sleep_time in zip([queue1, queue2], [engine1, engine2], [0.1, 0.1])
        ]

        # Wait until all tasks finish
        boards, results1, results2 = await asyncio.gather(
            data_generator_task,
            data_consumer_task1,
            data_consumer_task2
            # data_generator_task
        )

        # results1 = await data_consumer_task1.wait()
        # results2 = await data_consumer_task2.wait()
        """

        # Wait until all tasks finish
        boards, results1, results2 = await asyncio.gather(data_generator_task, data_consumer_task1)
        """
        for board_index, board in enumerate(boards):
            best_move1, move_stats1, _ = results1[board_index]
            best_move2, move_stats2, _ = results2[board_index]

            if best_move1 != "invalid" and best_move2 != "invalid":
                results.append((move_stats1[best_move1].Q, move_stats2[best_move2].Q))
                boards_final.append(board)

    return boards_final, results


def analyze_results(
    boards: List[chess.Board],
    results: List[Tuple[float, float]],
    difference_threshold: float = 1.0,
    show: bool = True,
) -> None:
    # Find results where the models have a large difference
    results = np.array(results)
    differences = np.abs(results[:, 0] - results[:, 1])
    indices = np.where(differences >= difference_threshold)[0]

    special_results = []

    for index in indices:
        board = boards[index]
        value1, value2 = results[index]
        special_results.append((board, differences[index], value1, value2))
        if show:
            title = f"value1: {value1}, value2: {value2}"
            plot_board(board, title=title, fen=board.fen(en_passant="fen"))

    return special_results


if __name__ == "__main__":
    ##################################
    #           CONFIG START         #
    ##################################
    SEED = 42
    ENGINE_CONFIG_NAME = "default_remote.ini"  # "remote_400_nodes.ini"
    DATA_CONFIG_NAME = "random_many_pieces.ini"
    POSITIONS = []
    NUM_POSITIONS = 100
    DIFFERENCE_THRESHOLD = 1
    NETWORK_PATH1 = "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    NETWORK_PATH2 = "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    ##################################
    #           CONFIG END           #
    ##################################

    np.random.seed(SEED)
    RemoteEngineConfig.set_config_folder_path(
        Path(__file__).parent.absolute() / Path("configs/engine_configs/")
    )
    BoardGeneratorConfig.set_config_folder_path(
        Path(__file__).parent.absolute() / Path("configs/data_generator_configs")
    )
    engine_config = RemoteEngineConfig.from_config_file(ENGINE_CONFIG_NAME)
    data_config = BoardGeneratorConfig.from_config_file(DATA_CONFIG_NAME)
    data_generator = RandomBoardGenerator(**data_config.board_generator_config)

    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    boards, results = asyncio.run(
        differential_testing(
            remote_host=engine_config.remote_host,
            remote_user=engine_config.remote_user,
            password_required=engine_config.password_required,
            engine_path=engine_config.engine_path,
            engine_config=engine_config.engine_config,
            network_path1=str(Path(engine_config.network_base_path) / Path(NETWORK_PATH1)),
            network_path2=str(Path(engine_config.network_base_path) / Path(NETWORK_PATH2)),
            data_generator=data_generator,
            search_limits=engine_config.search_limits,
            num_positions=NUM_POSITIONS,
        )
    )
    print("Results = ", results)

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time: .3f} seconds")
    if results:
        special_cases = analyze_results(
            boards, results, difference_threshold=DIFFERENCE_THRESHOLD, show=True
        )
        print(
            f"Found {len(special_cases)} boards where "
            f"evaluation differs by more than {DIFFERENCE_THRESHOLD}!"
        )
        for board, difference, v1, v2 in special_cases:
            fen = board.fen(en_passant="fen")
            print(f"{fen}, {difference = :.2f}, {v1 = :.3f}, {v2 = :.3f}")
