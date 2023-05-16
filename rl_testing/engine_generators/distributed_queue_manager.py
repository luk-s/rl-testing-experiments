from multiprocessing.managers import SyncManager
from typing import Tuple
import queue

address = "127.0.0.1"
port = 50000
password = "password"


class QueueManager(SyncManager):
    pass


def connect_to_manager() -> Tuple[queue.Queue, queue.Queue]:
    QueueManager.register("input_queue")
    QueueManager.register("output_queue")
    manager = QueueManager(address=(address, port), authkey=password.encode("utf-8"))
    manager.connect()
    return manager.input_queue(), manager.output_queue()
