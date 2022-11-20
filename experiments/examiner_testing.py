import argparse
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.util.util import cp2q

RESULT_DIR = Path(__file__).parent / Path("results/examiner_testing")


async def create_positions(
    consumer_queues: List[asyncio.Queue],
    data_generator: BoardGenerator,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    identifier_str: str = "",
) -> None:

    # Create random chess positions if necessary
    for board_index in range(num_positions):

        # Create a random chess position
        board_candidate = data_generator.next()

        # Check if the generated position was valid
        if board_candidate != "failed":
            fen = board_candidate.fen(en_passant="fen")
            print(f"[{identifier_str}] Created board {board_index}: " f"{fen}")
            for queue in consumer_queues:
                await queue.put(board_candidate.copy())

            await asyncio.sleep(delay=sleep_between_positions)


async def create_candidates(
    victim_pop_queue: asyncio.Queue,
    victim_push_queue: asyncio.Queue,
    examiner_pop_queue: asyncio.Queue,
    data_pop_queue: asyncio.Queue,
    data_push_queue: asyncio.Queue,
    threshold_correct: float,
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    board_counter = 1
    fen_cache = {}

    while True:
        # Fetch the next board and the corresponding scores from the queues
        board = await data_pop_queue.get()
        board_v, score_v = await victim_pop_queue.get()
        board_e, score_e = await examiner_pop_queue.get()
        assert board.fen() == board_v.fen() == board_e.fen()
        await asyncio.sleep(delay=sleep_after_get)

        print(f"[{identifier_str}] Received board {board_counter}: " + board.fen(en_passant="fen"))

        # Check if the board is promising
        if (
            score_v != "invalid"
            and score_e != "invalid"
            and np.abs(score_v - score_e) <= threshold_correct
        ):
            print(
                f"[{identifier_str}] Created candidate board {board_counter}: "
                + board.fen(en_passant="fen")
            )

            original_board = board.copy()

            # Create potential adversarial examples
            legal_moves1 = board.legal_moves
            for move1 in legal_moves1:
                board.push(move1)
                legal_moves2 = board.legal_moves
                for move2 in legal_moves2:
                    board.push(move2)

                    # Avoid analyzing the same position twice
                    fen = board.fen(en_passant="fen")
                    if fen in fen_cache or board.legal_moves.count() == 0:
                        board.pop()
                        continue
                    fen_cache[fen] = True

                    # Analyze the potential adversarial example
                    await victim_push_queue.put(board.copy())
                    await data_push_queue.put(
                        (original_board, board.copy(), score_v, score_e, move1, move2)
                    )

                    # Undo the moves
                    board.pop()
                board.pop()

        else:
            reason = ""
            if score_v == "invalid":
                reason += "score_v is invalid, "
            if score_e == "invalid":
                reason += "score_e is invalid, "
            if np.abs(score_v - score_e) > threshold_correct:
                reason += f"{score_v = } and {score_e = } are too different, "
            print(
                f"[{identifier_str}] Board {board_counter}"
                + board.fen(en_passant="fen")
                + " is not promising: "
                + reason
            )
        board_counter += 1

        # Mark the board as processed
        victim_pop_queue.task_done()
        examiner_pop_queue.task_done()
        data_pop_queue.task_done()


async def filter_candidates(
    victim_pop_queue: asyncio.Queue,
    examiner_push_queue: asyncio.Queue,
    data_pop_queue: asyncio.Queue,
    data_push_queue: asyncio.Queue,
    threshold_adversarial: float,
    threshold_equal: float,
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    board_counter = 1
    while True:
        # Fetch the next board and the corresponding scores from the queues
        (
            board_original,
            board_adversarial,
            score_original_v,
            score_original_e,
            move1,
            move2,
        ) = await data_pop_queue.get()
        board_v, score_v = await victim_pop_queue.get()
        assert board_adversarial.fen() == board_v.fen()
        await asyncio.sleep(delay=sleep_after_get)

        print(
            f"[{identifier_str}] Received candidate {board_counter}: "
            + board_adversarial.fen(en_passant="fen")
        )

        # Check if the board is promising
        if score_v != "invalid" and np.abs(score_v - score_original_e) > np.abs(
            threshold_adversarial - threshold_equal
        ):
            print(
                f"[{identifier_str}] Candidate board {board_counter}: "
                + board_adversarial.fen(en_passant="fen")
                + " is promising!"
            )

            await examiner_push_queue.put(board_adversarial.copy())
            await data_push_queue.put(
                (
                    board_original,
                    board_adversarial,
                    score_original_v,
                    score_original_e,
                    score_v,
                    move1,
                    move2,
                )
            )
        else:
            reason = ""
            if score_v == "invalid":
                reason += "score_v is invalid, "
            if np.abs(score_v - score_original_e) <= np.abs(
                threshold_adversarial - threshold_equal
            ):
                reason += f"{score_v = } and {score_original_e = } are too similar, "
            print(
                f"[{identifier_str}] Board {board_counter}: "
                + board_adversarial.fen(en_passant="fen")
                + " is not promising: "
                + reason
            )
        board_counter += 1

        # Mark the board as processed
        victim_pop_queue.task_done()
        data_pop_queue.task_done()


async def evaluate_candidates(
    examiner_pop_queue: asyncio.Queue,
    data_pop_queue: asyncio.Queue,
    threshold_adversarial: float,
    threshold_equal: float,
    file_path: Union[str, Path],
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    # Create a file to store the results
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    board_counter = 1

    with open(file_path, "a") as file:
        # Write the header
        file.write(
            "board_original,board_adversarial,score_original_v,score_original_e,"
            "score_adversarial_v,score_adversarial_e,move1,move2,success\n"
        )
        while True:
            # Fetch the next board and the corresponding scores from the queues
            (
                board_original,
                board_adversarial,
                score_original_v,
                score_original_e,
                score_adversarial_v,
                move1,
                move2,
            ) = await data_pop_queue.get()
            board_e, score_adversarial_e = await examiner_pop_queue.get()
            assert board_adversarial.fen() == board_e.fen()
            await asyncio.sleep(delay=sleep_after_get)

            # Check if the board is actually an adversarial example
            if (
                np.abs(score_adversarial_v - score_adversarial_e) >= threshold_adversarial
                and np.abs(score_original_e - score_adversarial_e) <= threshold_equal
            ):
                print(
                    f"[{identifier_str}] Found adversarial example! Board {board_counter}: "
                    + board_original.fen(en_passant="fen")
                    + f" ({move1}, {move2})"
                )
                success = 1
            else:
                reason = ""
                if np.abs(score_adversarial_v - score_adversarial_e) < threshold_adversarial:
                    reason += "score_adversarial_v and score_adversarial_e are too similar, "
                if np.abs(score_original_e - score_adversarial_e) > threshold_equal:
                    reason += "score_original_e and score_adversarial_e are too different, "
                print(
                    f"[{identifier_str}] Board {board_counter}: "
                    + board_adversarial.fen(en_passant="fen")
                    + " is not an adversarial example: "
                    + reason
                )
                success = 0
                # Write the found adversarial example into a file
            file.write(
                f"{board_original.fen()},"
                f"{board_adversarial.fen()},"
                f"{score_original_v},"
                f"{score_original_e},"
                f"{score_adversarial_v},"
                f"{score_adversarial_e},"
                f"{move1},"
                f"{move2},"
                f"{success}\n"
            )
            board_counter += 1

            # Mark the board as processed
            examiner_pop_queue.task_done()
            data_pop_queue.task_done()


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

    # Initialize the engine
    if network_name is not None:
        engine_generator.set_network(network_name=network_name)
    engine = await engine_generator.get_initialized_engine()

    while True:
        # Fetch the next board from the queue
        board = await consumer_queue.get()
        await asyncio.sleep(delay=sleep_after_get)

        print(
            f"[{identifier_str}] Analyzing board {board_counter}: " + board.fen(en_passant="fen")
        )
        try:
            # Analyze the board
            info = await engine.analyse(board, chess.engine.Limit(**search_limits))
        except chess.engine.EngineTerminatedError:
            if engine_generator is None:
                print("Can't restart engine due to missing generator")
                raise

            # Try to restart the engine
            print("Trying to restart engine")

            if network_name is not None:
                engine_generator.set_network(network_name=network_name)
            engine = await engine_generator.get_initialized_engine()

            # Add an error to the receiver queue
            await producer_queue.put((board, "invalid"))
        else:
            # Add the board to the receiver queue
            # The 12800 is used as maximum value because we use the q2cp function
            # to convert q_values to centipawns. This formula has values in
            # [-12800, 12800] for q_values in [-1, 1]
            await producer_queue.put((board, cp2q(info["score"].relative.score(mate_score=12800))))
        finally:
            consumer_queue.task_done()
            board_counter += 1


async def examiner_testing(
    victim_engine_generator: EngineGenerator,
    examiner_engine_generator: EngineGenerator,
    victim_network_name: Optional[str],
    examiner_network_name: Optional[str],
    data_generator: BoardGenerator,
    threshold_correct: float,
    threshold_equal: float,
    threshold_adversarial: float,
    *,
    search_limits_victim: Optional[Dict[str, Any]] = None,
    search_limits_examiner: Optional[Dict[str, Any]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    num_positions: int = 1,
    sleep_after_get: float = 0.1,
    queue_max_size: int = 10000,
) -> None:

    # Build the result file path
    if result_file_path is None:
        result_file_path = Path("results") / f"{victim_network_name}-{examiner_network_name}.csv"

    # Create all required queues
    victim_queue_original_in = asyncio.Queue(maxsize=queue_max_size)
    victim_queue_original_out = asyncio.Queue(maxsize=queue_max_size)
    victim_queue_adversarial_in = asyncio.Queue(maxsize=queue_max_size)
    victim_queue_adversarial_out = asyncio.Queue(maxsize=queue_max_size)
    examiner_queue_original_in = asyncio.Queue(maxsize=queue_max_size)
    examiner_queue_original_out = asyncio.Queue(maxsize=queue_max_size)
    examiner_queue_adversarial_in = asyncio.Queue(maxsize=queue_max_size)
    examiner_queue_adversarial_out = asyncio.Queue(maxsize=queue_max_size)
    data_queue1 = asyncio.Queue(maxsize=queue_max_size)
    data_queue2 = asyncio.Queue(maxsize=queue_max_size)
    data_queue3 = asyncio.Queue(maxsize=queue_max_size)

    # Create all data processing tasks
    data_generator_task = asyncio.create_task(
        create_positions(
            consumer_queues=[data_queue1, victim_queue_original_in, examiner_queue_original_in],
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_generator_task = asyncio.create_task(
        create_candidates(
            victim_pop_queue=victim_queue_original_out,
            victim_push_queue=victim_queue_adversarial_in,
            examiner_pop_queue=examiner_queue_original_out,
            data_pop_queue=data_queue1,
            data_push_queue=data_queue2,
            threshold_correct=threshold_correct,
            sleep_after_get=sleep_after_get,
            identifier_str="CANDIDATE_GENERATOR",
        )
    )

    candidate_filter_task = asyncio.create_task(
        filter_candidates(
            victim_pop_queue=victim_queue_adversarial_out,
            examiner_push_queue=examiner_queue_adversarial_in,
            data_pop_queue=data_queue2,
            data_push_queue=data_queue3,
            threshold_adversarial=threshold_adversarial,
            threshold_equal=threshold_equal,
            sleep_after_get=sleep_after_get,
            identifier_str="CANDIDATE_FILTER",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            examiner_pop_queue=examiner_queue_adversarial_out,
            data_pop_queue=data_queue3,
            threshold_adversarial=threshold_adversarial,
            threshold_equal=threshold_equal,
            file_path=result_file_path,
            sleep_after_get=sleep_after_get,
            identifier_str="CANDIDATE_EVALUATION",
        )
    )

    # Create all analysis tasks
    victim_original_analysis_task = asyncio.create_task(
        analyze_position(
            consumer_queue=victim_queue_original_in,
            producer_queue=victim_queue_original_out,
            search_limits=search_limits_victim,
            engine_generator=victim_engine_generator,
            network_name=victim_network_name,
            sleep_after_get=sleep_after_get,
            identifier_str="VICTIM_ORIGINAL_ANALYSIS",
        )
    )

    victim_adversarial_analysis_task = asyncio.create_task(
        analyze_position(
            consumer_queue=victim_queue_adversarial_in,
            producer_queue=victim_queue_adversarial_out,
            search_limits=search_limits_victim,
            engine_generator=victim_engine_generator,
            network_name=victim_network_name,
            sleep_after_get=sleep_after_get,
            identifier_str="VICTIM_ADVERSARIAL_ANALYSIS",
        )
    )

    examiner_original_analysis_task = asyncio.create_task(
        analyze_position(
            consumer_queue=examiner_queue_original_in,
            producer_queue=examiner_queue_original_out,
            search_limits=search_limits_examiner,
            engine_generator=examiner_engine_generator,
            network_name=examiner_network_name,
            sleep_after_get=sleep_after_get,
            identifier_str="EXAMINER_ORIGINAL_ANALYSIS",
        )
    )

    examiner_adversarial_analysis_task = asyncio.create_task(
        analyze_position(
            consumer_queue=examiner_queue_adversarial_in,
            producer_queue=examiner_queue_adversarial_out,
            search_limits=search_limits_examiner,
            engine_generator=examiner_engine_generator,
            network_name=examiner_network_name,
            sleep_after_get=sleep_after_get,
            identifier_str="EXAMINER_ADVERSARIAL_ANALYSIS",
        )
    )

    # Wait for data generator task to finish
    await asyncio.wait([data_generator_task])

    # Wait for data queues to be empty
    await data_queue1.join()
    await data_queue2.join()
    await data_queue3.join()

    # Cancel all remaining tasks
    candidate_generator_task.cancel()
    candidate_filter_task.cancel()
    candidate_evaluation_task.cancel()
    victim_original_analysis_task.cancel()
    victim_adversarial_analysis_task.cancel()
    examiner_original_analysis_task.cancel()
    examiner_adversarial_analysis_task.cancel()


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
    parser.add_argument("--victim_engine_config_name",      type=str, default="local_400_nodes.ini")  # noqa: E501
    parser.add_argument("--examiner_engine_config_name",    type=str, default="local_25_depth_stockfish.ini")  # noqa: E501
    # parser.add_argument("--victim_engine_config_name",      type=str, default="remote_400_nodes.ini")  # noqa: E501
    # parser.add_argument("--examiner_engine_config_name",    type=str, default="remote_25_depth_stockfish.ini")  # noqa: E501
    parser.add_argument("--data_config_name",               type=str, default="late_move_fen_database.ini")  # noqa: E501
    parser.add_argument("--num_positions",                  type=int, default=100_000)  # noqa: E501
    # parser.add_argument("--num_positions",                  type=int, default=100)  # noqa: E501
    parser.add_argument("--victim_network_path",            type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa: E501
    parser.add_argument("--examiner_network_path",          type=str, default="None")  # noqa: E501    
    parser.add_argument("--threshold_equal",                type=float, default=0.2)  # noqa: E501
    parser.add_argument("--threshold_correct",              type=float, default=0.3)  # noqa: E501
    parser.add_argument("--threshold_adversarial",          type=float, default=1)  # noqa: E501
    parser.add_argument("--queue_max_size",                 type=int, default=10000)  # noqa: E501
    parser.add_argument("--result_subdir",                  type=str, default="main_experiment")  # noqa: E501
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################
    args = parser.parse_args()

    np.random.seed(args.seed)
    config_folder_path = Path(__file__).parent.absolute() / Path("configs/engine_configs/")

    # Build the victim engine generator
    engine_config_victim = get_engine_config(
        config_name=args.victim_engine_config_name, config_folder_path=config_folder_path
    )
    engine_generator_victim = get_engine_generator(engine_config_victim)

    # Build the examiner engine generator
    engine_config_examiner = get_engine_config(
        config_name=args.examiner_engine_config_name, config_folder_path=config_folder_path
    )
    engine_generator_examiner = get_engine_generator(engine_config_examiner)

    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Create results-file-name
    victim_engine_config_name = args.victim_engine_config_name[:-4]
    examiner_engine_config_name = args.examiner_engine_config_name[:-4]
    data_config_name = args.data_config_name[:-4]

    # Build the result file path
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_VICTIM_ENGINE_{victim_engine_config_name}_"
        f"EXAMINER_ENGINE_{examiner_engine_config_name}_DATA_{data_config_name}.txt"
    )

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        examiner_testing(
            victim_network_name=args.victim_network_path
            if args.victim_network_path != "None"
            else None,
            examiner_network_name=args.examiner_network_path
            if args.examiner_network_path != "None"
            else None,
            victim_engine_generator=engine_generator_victim,
            examiner_engine_generator=engine_generator_examiner,
            data_generator=data_generator,
            threshold_correct=args.threshold_correct,
            threshold_equal=args.threshold_equal,
            threshold_adversarial=args.threshold_adversarial,
            search_limits_victim=engine_config_victim.search_limits,
            search_limits_examiner=engine_config_examiner.search_limits,
            result_file_path=result_file_path,
            num_positions=args.num_positions,
            queue_max_size=args.queue_max_size,
            sleep_after_get=0.1,
        )
    )

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time: .3f} seconds")
