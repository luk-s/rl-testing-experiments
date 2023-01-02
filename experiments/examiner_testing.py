import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.util.util import cp2q, get_task_result_handler
from rl_testing.util.experiment import store_experiment_params

RESULT_DIR = Path(__file__).parent / Path("results/examiner_testing")


class ReceiverCache:
    def __init__(self, queues: List[asyncio.Queue]) -> None:
        # Assumes that each queue returns a tuple of elements where the
        # first element is an instance of chess.Board
        assert len(queues) > 0
        assert all([isinstance(queue, asyncio.Queue) for queue in queues])
        self.queues = queues
        self.num_queues = len(queues)

        self.receiver_cache = {}

    async def receive_data(self) -> List[Iterable[Any]]:
        fens = []
        data = []

        # Receive data from all queues
        for queue in self.queues:
            all_data = await queue.get()

            if isinstance(all_data, list) or isinstance(all_data, tuple):
                board = all_data[0]
                info = all_data
                if len(info) == 1:
                    info = info[0]
            else:
                board = all_data
                info = all_data

            assert isinstance(board, chess.Board)

            # The boards might not arrive in the correct order due to the asynchronous nature of
            # the program. Therefore, we need to cache the boards and scores until we have all
            # of them.
            # # Prepare the cache entries for the current boards
            fens.append(board.fen())
            data.append(info)

        indices = list(range(self.num_queues))

        complete_data_tuples = []

        # Put the data into the cache
        for fen, info, index in zip(fens, data, indices):
            if fen in self.receiver_cache:
                self.receiver_cache[fen][index] = info
            else:
                self.receiver_cache[fen] = [None] * self.num_queues
                self.receiver_cache[fen][index] = info

            # Check if we have all the data for this board
            if all([element is not None for element in self.receiver_cache[fen]]):
                # We have all the data for this board
                complete_data_tuples.append(self.receiver_cache[fen])
                del self.receiver_cache[fen]

        return complete_data_tuples


async def create_positions(
    data_queue: asyncio.Queue,
    analysis_queue_tuples: List[Tuple[asyncio.PriorityQueue, asyncio.Queue, str]],
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
            logging.info(
                f"[{identifier_str}] Created board {board_index + 1}: " f"{fen}"
            )
            await data_queue.put(board_candidate.copy())
            for queue_in, queue_out, queue_identifier_str in analysis_queue_tuples:
                # Assign low priority to the new boards (0 = high, 1 = low)
                # This ensures that the whole pipeline is emptied before new boards are created
                # and that no thread is starved.
                # Use the current ime as second argument to ensure that the priority queue is
                # sorted by time if the priority is the same.

                await queue_in.put(
                    (
                        1,
                        time.time(),
                        (board_candidate.copy(), queue_out, queue_identifier_str),
                    )
                )

            await asyncio.sleep(delay=sleep_between_positions)


async def create_candidates(
    victim_pop_queue: asyncio.Queue,
    victim_push_queue_tuple: Tuple[asyncio.PriorityQueue, asyncio.Queue, str],
    examiner_pop_queue: asyncio.Queue,
    data_pop_queue: asyncio.Queue,
    data_push_queue: asyncio.Queue,
    threshold_correct: float,
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    board_counter = 1
    fen_cache = {}

    # The order of the queues is important! The 'receive' function will return the data in the
    # same order as the queues are given to the initializer.
    receiver_cache = ReceiverCache(
        [data_pop_queue, victim_pop_queue, examiner_pop_queue]
    )

    while True:
        # Fetch the next board and the corresponding scores from the queues
        complete_data_tuples = await receiver_cache.receive_data()

        # Create the candidate boards for the complete data tuples
        for board, (_, score_v), (_, score_e) in complete_data_tuples:
            await asyncio.sleep(delay=sleep_after_get)

            logging.info(
                f"[{identifier_str}] Received board {board_counter}: "
                + board.fen(en_passant="fen")
            )

            # Check if the board is promising
            if (
                score_v != "invalid"
                and score_e != "invalid"
                and np.abs(score_v - score_e) <= threshold_correct
            ):
                logging.info(
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
                        # Assign high priority to these boards (0 = high, 1 = low)
                        # This ensures that the whole pipeline is emptied before new boards are
                        # created and that no thread is starved.
                        (
                            victim_push_queue_in,
                            victim_push_queue_out,
                            queue_identifier_str,
                        ) = victim_push_queue_tuple
                        await victim_push_queue_in.put(
                            (
                                0,
                                time.time(),
                                (
                                    board.copy(),
                                    victim_push_queue_out,
                                    queue_identifier_str,
                                ),
                            )
                        )
                        await data_push_queue.put(
                            (
                                board.copy(),
                                original_board,
                                score_v,
                                score_e,
                                move1,
                                move2,
                            )
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
                logging.info(
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
    examiner_push_queue_tuple: Tuple[asyncio.PriorityQueue, asyncio.Queue, str],
    data_pop_queue: asyncio.Queue,
    data_push_queue: asyncio.Queue,
    threshold_adversarial: float,
    threshold_equal: float,
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    board_counter = 1

    # The order of the queues is important! The 'receive' function will return the data in the
    # same order as the queues are given to the initializer.
    receiver_cache = ReceiverCache([data_pop_queue, victim_pop_queue])

    while True:
        # Fetch the next board and the corresponding scores from the queues
        complete_data_tuples = await receiver_cache.receive_data()

        # Iterate over the received data tuples
        for (
            board_adversarial,
            board_original,
            score_original_v,
            score_original_e,
            move1,
            move2,
        ), (_, score_v) in complete_data_tuples:

            await asyncio.sleep(delay=sleep_after_get)

            logging.info(
                f"[{identifier_str}] Received candidate {board_counter}: "
                + board_adversarial.fen(en_passant="fen")
            )

            # Check if the board is promising
            if score_v != "invalid" and np.abs(score_v - score_original_e) >= np.abs(
                threshold_adversarial - threshold_equal
            ):
                logging.info(
                    f"[{identifier_str}] Candidate board {board_counter}: "
                    + board_adversarial.fen(en_passant="fen")
                    + " is promising!"
                )

                # Analyze the potential adversarial example with the examiner
                # Assign high priority to these boards (0 = high, 1 = low)
                # This ensures that the whole pipeline is emptied before new boards are created
                # and that no thread is starved.
                (
                    examiner_push_queue_in,
                    examiner_push_queue_out,
                    queue_identifier_str,
                ) = examiner_push_queue_tuple
                await examiner_push_queue_in.put(
                    (
                        0,
                        time.time(),
                        (
                            board_adversarial.copy(),
                            examiner_push_queue_out,
                            queue_identifier_str,
                        ),
                    )
                )
                await data_push_queue.put(
                    (
                        board_adversarial,
                        board_original,
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
                    reason += (
                        f"{score_v = } and {score_original_e = } are too similar, "
                    )
                logging.info(
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

    # The order of the queues is important! The 'receive' function will return the data in the
    # same order as the queues are given to the initializer.
    receiver_cache = ReceiverCache([data_pop_queue, examiner_pop_queue])

    with open(file_path, "a") as file:
        # Write the header
        file.write(
            "board_original,board_adversarial,score_original_v,score_original_e,"
            "score_adversarial_v,score_adversarial_e,move1,move2,success\n"
        )

        while True:
            # Fetch the next board and the corresponding scores from the queues
            complete_data_tuples = await receiver_cache.receive_data()

            # Iterate over the received data
            for (
                board_adversarial,
                board_original,
                score_original_v,
                score_original_e,
                score_adversarial_v,
                move1,
                move2,
            ), (_, score_adversarial_e) in complete_data_tuples:
                await asyncio.sleep(delay=sleep_after_get)

                # Check if the board is actually an adversarial example
                if (
                    np.abs(score_adversarial_v - score_adversarial_e)
                    >= threshold_adversarial
                    and np.abs(score_original_e - score_adversarial_e)
                    <= threshold_equal
                ):
                    logging.info(
                        f"[{identifier_str}] Found adversarial example! Board {board_counter}: "
                        + board_original.fen(en_passant="fen")
                        + f" ({move1}, {move2})"
                    )
                    success = 1
                else:
                    reason = ""
                    if (
                        np.abs(score_adversarial_v - score_adversarial_e)
                        < threshold_adversarial
                    ):
                        reason += (
                            f"{score_adversarial_v = } and {score_adversarial_e = } "
                            "are too similar, "
                        )
                    if np.abs(score_original_e - score_adversarial_e) > threshold_equal:
                        reason += (
                            f"{score_original_e = } and {score_adversarial_e = } "
                            "are too different, "
                        )
                    logging.info(
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
    consumer_queue: asyncio.PriorityQueue,
    search_limits: Dict[str, Any],
    engine_generator: EngineGenerator,
    network_name: Optional[str] = None,
    sleep_after_get: float = 0.1,
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
        # Fetch the next board, the producer queue and the identifier string from the queue
        _, _, (board, producer_queue, identifier_str) = await consumer_queue.get()
        await asyncio.sleep(delay=sleep_after_get)

        logging.info(
            f"[{identifier_str}] Analyzing board {board_counter}: "
            + board.fen(en_passant="fen")
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

            # Try to restart the engine
            logging.info("Trying to restart engine")

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
            await producer_queue.put(
                (board, cp2q(info["score"].relative.score(mate_score=12800)))
            )
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
    queue_max_size: int = 10000,
    num_victim_workers: int = 2,
    num_examiner_workers: int = 2,
    sleep_after_get: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> None:

    if logger is None:
        logger = logging.getLogger(__name__)

    # Build the result file path
    if result_file_path is None:
        result_file_path = (
            Path("results") / f"{victim_network_name}-{examiner_network_name}.csv"
        )

    # Create all required queues
    # Only use one input queue for the victim and examiner
    # to allow multiple workers to fetch from the same queue
    victim_queue_in = asyncio.PriorityQueue(maxsize=queue_max_size)
    victim_queue_original_out = asyncio.Queue(maxsize=queue_max_size)
    victim_queue_adversarial_out = asyncio.Queue(maxsize=queue_max_size)
    examiner_queue_in = asyncio.PriorityQueue(maxsize=queue_max_size)
    examiner_queue_original_out = asyncio.Queue(maxsize=queue_max_size)
    examiner_queue_adversarial_out = asyncio.Queue(maxsize=queue_max_size)
    data_queue1 = asyncio.Queue(maxsize=queue_max_size)
    data_queue2 = asyncio.Queue(maxsize=queue_max_size)
    data_queue3 = asyncio.Queue(maxsize=queue_max_size)

    # Create all data processing tasks
    data_generator_task = asyncio.create_task(
        create_positions(
            data_queue=data_queue1,
            analysis_queue_tuples=[
                (
                    victim_queue_in,
                    victim_queue_original_out,
                    "VICTIM_ORIGINAL_ANALYSIS",
                ),
                (
                    examiner_queue_in,
                    examiner_queue_original_out,
                    "EXAMINER_ORIGINAL_ANALYSIS",
                ),
            ],
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_generator_task = asyncio.create_task(
        create_candidates(
            victim_pop_queue=victim_queue_original_out,
            victim_push_queue_tuple=(
                victim_queue_in,
                victim_queue_adversarial_out,
                "VICTIM_ADVERSARIAL_ANALYSIS",
            ),
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
            examiner_push_queue_tuple=(
                examiner_queue_in,
                examiner_queue_adversarial_out,
                "EXAMINER_ADVERSARIAL_ANALYSIS",
            ),
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
    victim_analysis_tasks = [
        asyncio.create_task(
            analyze_position(
                consumer_queue=victim_queue_in,
                search_limits=search_limits_victim,
                engine_generator=victim_engine_generator,
                network_name=victim_network_name,
                sleep_after_get=sleep_after_get,
            )
        )
        for _ in range(num_victim_workers)
    ]

    examiner_analysis_tasks = [
        asyncio.create_task(
            analyze_position(
                consumer_queue=examiner_queue_in,
                search_limits=search_limits_examiner,
                engine_generator=examiner_engine_generator,
                network_name=examiner_network_name,
                sleep_after_get=sleep_after_get,
            )
        )
        for _ in range(num_examiner_workers)
    ]

    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    for task in (
        [
            data_generator_task,
            candidate_generator_task,
            candidate_filter_task,
            candidate_evaluation_task,
        ]
        + victim_analysis_tasks
        + examiner_analysis_tasks
    ):
        task.add_done_callback(handle_task_exception)

    # Wait for data generator task to finish
    await asyncio.wait([data_generator_task])

    # Wait for data queues to become empty
    await data_queue1.join()
    await data_queue2.join()
    await data_queue3.join()

    # Cancel all remaining tasks
    candidate_generator_task.cancel()
    candidate_filter_task.cancel()
    candidate_evaluation_task.cancel()
    for victim_analysis_task in victim_analysis_tasks:
        victim_analysis_task.cancel()
    for examiner_analysis_task in examiner_analysis_tasks:
        examiner_analysis_task.cancel()


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
    parser.add_argument("--data_config_name",               type=str, default="database.ini")  # noqa: E501
    parser.add_argument("--num_positions",                  type=int, default=100_000)  # noqa: E501
    # parser.add_argument("--num_positions",                  type=int, default=100)  # noqa: E501
    parser.add_argument("--victim_network_path",            type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa: E501
    parser.add_argument("--examiner_network_path",          type=str, default="None")  # noqa: E501
    parser.add_argument("--threshold_equal",                type=float, default=0.2)  # noqa: E501
    parser.add_argument("--threshold_correct",              type=float, default=0.3)  # noqa: E501
    parser.add_argument("--threshold_adversarial",          type=float, default=1)  # noqa: E501
    parser.add_argument("--queue_max_size",                 type=int, default=10000)  # noqa: E501
    parser.add_argument("--num_victim_workers",             type=int, default=2)  # noqa: E501
    parser.add_argument("--num_examiner_workers",           type=int, default=2)  # noqa: E501
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
    config_folder_path = Path(__file__).parent.absolute() / Path(
        "configs/engine_configs/"
    )

    # Build the victim engine generator
    engine_config_victim = get_engine_config(
        config_name=args.victim_engine_config_name,
        config_folder_path=config_folder_path,
    )
    engine_generator_victim = get_engine_generator(engine_config_victim)

    # Build the examiner engine generator
    engine_config_examiner = get_engine_config(
        config_name=args.examiner_engine_config_name,
        config_folder_path=config_folder_path,
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

    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
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
            num_victim_workers=args.num_victim_workers,
            num_examiner_workers=args.num_examiner_workers,
            sleep_after_get=0.1,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
