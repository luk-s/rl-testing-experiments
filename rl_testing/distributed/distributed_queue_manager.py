from multiprocessing.managers import SyncManager
from typing import Optional, Tuple
import queue

default_address = "127.0.0.1"
default_port = 50000
default_password = "password"


class QueueManager(SyncManager):
    pass


def connect_to_manager(
    address: str = default_address, port: str = default_port, password: str = default_password
) -> Tuple[queue.Queue, queue.Queue, Optional[str]]:
    QueueManager.register("input_queue")
    QueueManager.register("output_queue")
    QueueManager.register("required_engine_config_name")
    manager = QueueManager(address=(address, port), authkey=password.encode("utf-8"))
    manager.connect()
    return (
        manager.input_queue(),
        manager.output_queue(),
        manager.required_engine_config_name()._getvalue(),
    )
