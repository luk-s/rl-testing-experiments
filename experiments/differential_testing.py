import argparse
import asyncio
import logging
import os
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config
from rl_testing.data_generators import BoardGenerator, get_data_generator

from rl_testing.distributed.distributed_queue_manager import (
    default_address,
    connect_to_manager,
    default_password,
    default_port,
)
from rl_testing.distributed.queue_utils import build_manager, SocketAddress, EmptySocketAddress
from rl_testing.distributed.worker import AnalysisObject

from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import get_task_result_handler

RESULT_DIR = Path(__file__).parent / Path("results/differential_testing")
CONFIG_DIR = Path(__file__).parent.absolute() / Path("configs")

# Define extra types for type hints
FEN = str
Move = str
Score = float


class ReceiverCache:
    """This class is used to cache the received data until all the data for a board has been
    received.
    """

    def __init__(
        self,
        consumer_queue1: queue.Queue,
        consumer_queue2: queue.Queue,
        sleep_after_get: float = 0.01,
    ) -> None:
        """Initializes the ReceiverCache object.

        Args:
            consumer_queue (queue.Queue): A queue from which the data is received.
        """
        # Print type of consumer_queue
        self.consumer_queue1 = consumer_queue1
        self.consumer_queue2 = consumer_queue2
        self.sleep_after_get = sleep_after_get

        self.score_cache: Dict[FEN, Tuple[Tuple[Score, Move], Tuple[Score, Move]]] = {}

    async def receive_data(self) -> Tuple[FEN, Score, Move, Score, Move]:
        """This function repeatedly fetches data from the two queues until all the data for a single
        board has been received.

        Returns:
             Tuple[FEN, Score, Move, Score, Move]: A tuple containing the FEN of the board, and
                the scores and best moves as returned by the two engines.
        """
        # Try to receive data from one of the two queues
        data_received = False
        while not data_received:
            # We order the two queues randomly to avoid starvation
            queues = (
                [self.consumer_queue1, self.consumer_queue2]
                if np.random.rand() < 0.5
                else [self.consumer_queue2, self.consumer_queue1]
            )
            for consumer_queue in queues:
                try:
                    analysis_object: AnalysisObject = consumer_queue.get_nowait()
                    fen: FEN = analysis_object.fen
                    score: Score = analysis_object.score
                    best_move: Move = analysis_object.best_move
                except queue.Empty:
                    pass
                else:
                    data_received = True
                    data_index = 0 if consumer_queue == self.consumer_queue1 else 1
                    break
            else:
                await asyncio.sleep(self.sleep_after_get)
                continue

        # The boards might not arrive in the correct order due to the asynchronous nature of
        # the program. Therefore, we need to cache the boards and scores until we have both
        # of them.
        if fen not in self.score_cache:
            self.score_cache[fen] = [None, None]

        self.score_cache[fen][data_index] = (score, best_move)

        complete_data_tuples = []
        # Check if we have all the data for this board
        if None not in self.score_cache[fen]:
            # We have all the data for this board
            complete_data_tuples.append((fen, *self.score_cache[fen]))

            # This saves memory
            del self.score_cache[fen]

        return complete_data_tuples


async def create_positions(
    data_generator: BoardGenerator,
    num_positions: int = 1,
    socket_address1: SocketAddress = EmptySocketAddress,
    socket_address2: SocketAddress = EmptySocketAddress,
    sleep_between_positions: float = 0.01,
    identifier_str: str = "",
) -> None:
    """Create chess positions using the provided data generator, and send the results to the
    output queues.

    Args:
        data_generator (BoardGenerator): A BoardGenerator object that is used to create the
            chess positions.
        num_positions (int, optional): The number of chess positions to create. Defaults to 1.
        socket_address1 (SocketAddress, optional): An object that contains address information
            about the first queue. Defaults to EmptySocketAddress.
        socket_address2 (SocketAddress, optional): An object that contains address information
            about the second queue. Defaults to EmptySocketAddress.
        sleep_between_positions (float, optional): The number of seconds to wait between creating
            two chess positions. Useful to pause this async function and allow other async
            functions to run. Defaults to 0.01.
        identifier_str (str, optional): A string that is used to identify this process.
            Defaults to "".
    """
    fen_cache = set()

    # Get the queues
    output_queue1: queue.Queue
    output_queue2: queue.Queue
    output_queue1, _, _ = connect_to_manager(**socket_address1.to_dict())
    output_queue2, _, _ = connect_to_manager(**socket_address2.to_dict())

    # Create the chess positions
    board_index = 1
    while board_index <= num_positions:
        # Create a random chess position
        board_candidate = data_generator.next()
        fen = board_candidate.fen(en_passant="fen")

        # Check if the generated position was valid
        if board_candidate != "failed" and fen not in fen_cache:
            fen_cache.add(fen)

            logging.info(f"[{identifier_str}] Created board {board_index + 1}: {fen}")

            # Send the board to the output queue
            output_queue1.put(AnalysisObject(fen=fen))
            output_queue2.put(AnalysisObject(fen=fen))

            await asyncio.sleep(delay=sleep_between_positions)

            board_index += 1


async def evaluate_candidates(
    file_path: Union[str, Path],
    num_positions: int = 1,
    socket_address1: SocketAddress = EmptySocketAddress,
    socket_address2: SocketAddress = EmptySocketAddress,
    sleep_after_get: float = 0.01,
    identifier_str: str = "",
) -> None:
    """This function receives the evaluated chess positions from the input queue and writes them
    to a file.

    Args:
        file_path (Union[str, Path]): The path to the file in which the results are stored.
        num_positions (int, optional): The number of chess positions to evaluate. Defaults to 1.
        socket_address1 (SocketAddress, optional): An object that contains address information
            about the first queue. Defaults to EmptySocketAddress.
        socket_address2 (SocketAddress, optional): An object that contains address information
            about the second queue. Defaults to EmptySocketAddress.
        sleep_after_get (float, optional): The number of seconds to wait after receiving a board
            from the input queue. Useful to pause this async function and allow other async
            functions to run. Defaults to 0.01.
        identifier_str (str, optional): A string that is used to identify this process.
            Defaults to "".
    """
    # Get the queues
    engine_queue1: queue.Queue
    engine_queue2: queue.Queue
    _, engine_queue1, _ = connect_to_manager(**socket_address1.to_dict())
    _, engine_queue2, _ = connect_to_manager(**socket_address2.to_dict())

    # Create a file to store the results
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    board_counter = 1
    flush_every = 1000

    # Initialize the receiver cache
    receiver_cache = ReceiverCache(
        consumer_queue1=engine_queue1,
        consumer_queue2=engine_queue2,
        sleep_after_get=sleep_after_get,
    )

    with open(file_path, "a") as file:
        while board_counter <= num_positions:
            # Fetch the next board and the corresponding scores from the queues
            complete_data_tuples = await receiver_cache.receive_data()

            # Iterate over the received data
            for fen, (score1, best_move1), (score2, best_move2) in complete_data_tuples:
                logging.info(f"[{identifier_str}] Saving board {board_counter}: " + fen)

                # Write the found adversarial example into a file
                result_str = f"{fen},{score1},{best_move1},{score2},{best_move2}\n"

                # Write the result to the file
                file.write(result_str)

                if board_counter % flush_every == 0:
                    file.flush()
                    os.fsync(file.fileno())

                board_counter += 1
                engine_queue1.task_done()
                engine_queue2.task_done()


async def differential_testing(
    data_generator: BoardGenerator,
    *,
    result_file_path: Optional[Union[str, Path]] = None,
    num_positions: int = 1,
    required_engine_config1: Optional[str] = None,
    required_engine_config2: Optional[str] = None,
    socket_address1: SocketAddress = EmptySocketAddress,
    socket_address2: SocketAddress = EmptySocketAddress,
    sleep_after_get: float = 0.01,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Main function which starts all the asynchronous tasks and manages the distributed queues.

    Args:
        data_generator (BoardGenerator): A BoardGenerator object that is used to create the
            chess positions.
        result_file_path (Optional[Union[str, Path]], optional): The path to the file in which the
            results are stored. Defaults to None.
        num_positions (int, optional): The number of chess positions to create. Defaults to 1.
        required_engine_config1 (Optional[str], optional): The name of the first engine
            configuration which worker processes should use. Defaults to None.
        required_engine_config2 (Optional[str], optional): The name of the second engine
            configuration which worker processes should use. Defaults to None.
        socket_address1 (SocketAddress, optional): An object that contains address information
            about the first queue. Defaults to EmptySocketAddress.
        socket_address2 (SocketAddress, optional): An object that contains address information
            about the second queue. Defaults to EmptySocketAddress.
        sleep_after_get (float, optional): The number of seconds to wait after receiving a board
            from the input queue. Useful to pause async functions and allow other async
            functions to run. Defaults to 0.1.
        logger (Optional[logging.Logger], optional): A logger object that is used to log messages.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    assert result_file_path is not None, "Result file path must be specified"

    # Set up the two sets of distributed queues
    net_manager1 = build_manager(
        **socket_address1.to_dict(),
        required_engine_config=required_engine_config1,
    )
    net_manager2 = build_manager(
        **socket_address2.to_dict(),
        required_engine_config=required_engine_config2,
    )
    net_manager1.start()
    net_manager2.start()

    # Create all data processing tasks
    data_generator_task = asyncio.create_task(
        create_positions(
            data_generator=data_generator,
            num_positions=num_positions,
            socket_address1=socket_address1,
            socket_address2=socket_address2,
            sleep_between_positions=sleep_after_get,
            identifier_str="DATA_GENERATOR",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            file_path=result_file_path,
            num_positions=num_positions,
            socket_address1=socket_address1,
            socket_address2=socket_address2,
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

    # Wait for tasks to finish
    await asyncio.wait([data_generator_task, candidate_evaluation_task])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    parser.add_argument("--engine_config_name1",             type=str, default="leela_local_400_nodes.ini")  # noqa
    parser.add_argument("--engine_config_name2",             type=str, default="leela_local_400_nodes.ini")  # noqa
    parser.add_argument("--data_config_name",                type=str, default="database.ini")  # noqa
    parser.add_argument("--address1",                        type=str, default=default_address)  # noqa
    parser.add_argument("--port1",                           type=int, default=default_port)  # noqa
    parser.add_argument("--password1",                       type=str, default=default_password)  # noqa
    parser.add_argument("--address2",                        type=str, default=default_address)  # noqa
    parser.add_argument("--port2",                           type=int, default=default_port)  # noqa
    parser.add_argument("--password2",                       type=str, default=default_password)  # noqa
    parser.add_argument("--num_positions",                   type=int, default=1_000_000)  # noqa
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

    # Build the data generator
    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=CONFIG_DIR / Path("data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Create results-file-name
    data_config_name = args.data_config_name[:-4]

    # Store current date and time as string
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")

    result_file_path = RESULT_DIR / Path(
        f"results_ENGINE1_{args.engine_config_name1[:-4]}_ENGINE2_{args.engine_config_name2}"
        f"_DATA_{data_config_name}_{dt_string}.txt"
    )

    # Store the experiment configuration in the result file
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Store the transformation names in the result file
    with open(result_file_path, "a") as result_file:
        result_file.write("fen,score1,best_move1,score2,best_move2\n")

    # Build the two address sockets
    socket_address1 = SocketAddress(address=args.address1, port=args.port1, password=args.password1)
    socket_address2 = SocketAddress(address=args.address2, port=args.port2, password=args.password2)

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        differential_testing(
            data_generator=data_generator,
            result_file_path=result_file_path,
            num_positions=args.num_positions,
            required_engine_config1=args.engine_config_name1,
            required_engine_config2=args.engine_config_name2,
            socket_address1=socket_address1,
            socket_address2=socket_address2,
            sleep_after_get=0.01,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
