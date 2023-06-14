import argparse
import queue
from rl_testing.distributed.distributed_queue_manager import (
    QueueManager,
    default_address,
    default_port,
    default_password,
)
from typing import Optional
import os
from datetime import datetime
from pathlib import Path
import signal


def build_manager(
    required_engine_config: Optional[str] = None,
    address: str = default_address,
    port: str = default_port,
    password: str = default_password,
) -> QueueManager:
    # Set up the distributed queues
    engine_queue_in: queue.Queue = queue.Queue()
    engine_queue_out: queue.Queue = queue.Queue()

    def get_input_queue() -> queue.Queue:
        return engine_queue_in

    def get_output_queue() -> queue.Queue:
        return engine_queue_out

    def get_required_engine_config() -> Optional[str]:
        return required_engine_config

    # Initialize the input- and output queues
    QueueManager.register("required_engine_config_name", callable=get_required_engine_config)
    QueueManager.register("input_queue", callable=get_input_queue)
    QueueManager.register("output_queue", callable=get_output_queue)

    net_manager = QueueManager(address=(address, port), authkey=password.encode("utf-8"))

    return net_manager


def start_queue(
    required_engine_config: Optional[str] = None,
    address: str = default_address,
    port: str = default_port,
    password: str = default_password,
) -> None:
    """Starts a queue server.

    Args:
        required_engine_config (Optional[str], optional): The name of the engine config this
            queue requires. Defaults to None.
        address (str, optional): The address of the queue. Defaults to default_address.
        port (str, optional): The port of the queue. Defaults to default_port.
        password (str, optional): The password of the queue. Defaults to default_password.

    Returns:
        Never returns. Runs in an infinite loop.
    """
    # Get the PID of the server and write it to a file
    pid = os.getpid()

    # Get the queue server
    net_manager = build_manager(
        required_engine_config=required_engine_config,
        address=address,
        port=port,
        password=password,
    )

    # Start the server
    net_manager.start()

    # Create a function which will terminate the net_manager
    # upon receiving a SIGINT
    def shutdown_handler(sig, frame):
        net_manager.shutdown()
        print(f"Shutting down queue with PID {pid}")
        exit(0)

    # Upon receiving a SIGIT or SIGTERM, terminate the net_manager
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Get the name of the folder of the current file
    current_folder = Path(__file__).parent.absolute()
    file_path = current_folder / ".queue_pids.txt"

    # Get nice string version of the current date and time
    now = datetime.now()
    now_str = now.strftime("%Y/%m/%d %H:%M:%S")

    with open(file_path, "a") as f:
        f.write(f"[{now_str}]: {pid}\n")

    print(f"Started queue with PID {pid}")

    # Sleep until the queue is terminated
    signal.pause()


def kill_queues() -> None:
    """
    Kills all the currently running queues.
    """
    # Get the name of the folder of the current file
    current_folder = Path(__file__).parent.absolute()

    file_path = current_folder / ".queue_pids.txt"

    if file_path.exists():
        # Kill all the queues
        with open(file_path, "r") as f:
            for line in f:
                pid = int(line.split("]:")[1])

                # Send a SIGTERM to the process
                os.kill(pid, signal.SIGTERM)

        # Remove the file
        os.system(f"rm {file_path}")

        print("Killed all queues")
    else:
        print("No queues to kill")


class QueueInterface:
    def __init__(
        self,
        required_engine_config: Optional[str] = None,
        address: str = default_address,
        port: str = default_port,
        password: str = default_password,
    ) -> None:
        self.engine_config_name = required_engine_config
        self.net_manager = build_manager(
            required_engine_config=required_engine_config,
            address=address,
            port=port,
            password=password,
        )

        self.net_manager.start()

        self.input_queue: queue.Queue = self.net_manager.input_queue()
        self.output_queue: queue.Queue = self.net_manager.output_queue()

    def __del__(self) -> None:
        self.net_manager.shutdown()

    def send(self, obj: object) -> None:
        self.input_queue.put(obj)

    def receive(self) -> object:
        return self.output_queue.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    parser.add_argument("--mode",                  type=str, default="start_queue", choices=["start_queue", "kill_queues"]) # noqa
    parser.add_argument("--engine_config_name",    type=str, default="local_400_nodes.ini")  # noqa
    parser.add_argument("--address",               type=str, default=default_address) # noqa
    parser.add_argument("--port",                  type=int, default=default_port) # noqa
    parser.add_argument("--password",              type=str, default=default_password) # noqa
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################

    args = parser.parse_args()

    if args.mode == "start_queue":
        start_queue(
            required_engine_config=args.engine_config_name,
            address=args.address,
            port=args.port,
            password=args.password,
        )

    elif args.mode == "kill_queues":
        kill_queues()

    else:
        raise ValueError(f"Unknown mode {args.mode}")
