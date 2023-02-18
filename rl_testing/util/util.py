import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.svg
import numpy as np


def get_task_result_handler(
    logger: logging.Logger,
    message: str,
    message_args: Tuple[Any, ...] = (),
) -> Any:
    def handle_task_result(
        task: asyncio.Task,
        *,
        logger: logging.Logger,
        message: str,
        message_args: Tuple[Any, ...] = (),
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Task cancellation should not be logged as an error.
        # Ad the pylint ignore: we want to handle all exceptions here so that the result of the task
        # is properly logged. There is no point re-raising the exception in this callback.
        except Exception:  # pylint: disable=broad-except
            logger.exception(message, *message_args)

    return lambda task: handle_task_result(
        task, logger=logger, message=message, message_args=message_args
    )


def get_random_state(
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.random.Generator:
    """Get a random state. Use the provided random state if it is not None, otherwise use the default random state.

    Args:
        random_state (Optional[Union[int, np.random.Generator]], optional): An optional random state or a seed. Defaults to None.

    Returns:
        np.random.Generator: The random state.
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, int):
        return np.random.default_rng(random_state)
    return random_state


if __name__ == "__main__":
    fen = "rnbqkbnr/pppp1ppp/8/3Pp3/5B2/6N1/PPP1PPPP/RNBQK2R w Kq e6 0 1"
    fen = "8/1p3r2/P1pK4/P3Pnp1/PpnR3N/Q4bqB/p2Bp3/Nk6 w - - 70 50"
    board = chess.Board(fen)
