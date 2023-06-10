from multiprocessing.managers import SyncManager
from typing import Optional, Tuple
import queue

default_address = "127.0.0.1"
default_port = 50000
default_password = "password"


class QueueManager(SyncManager):
    required_engine_config_name: Optional[str] = None

    def requires_engine_config(self) -> bool:
        return self.required_engine_config_name is not None

    @property
    def engine_config_name(self) -> Optional[str]:
        return QueueManager.required_engine_config_name

    @staticmethod
    def set_engine_config(engine_config: str) -> None:
        QueueManager.required_engine_config_name = engine_config


def connect_to_manager(
    address: str = default_address, port: str = default_port, password: str = default_password
) -> Tuple[queue.Queue, queue.Queue, Optional[str]]:
    QueueManager.register("input_queue")
    QueueManager.register("output_queue")
    manager = QueueManager(address=(address, port), authkey=password.encode("utf-8"))
    manager.connect()
    return manager.input_queue(), manager.output_queue(), manager.engine_config_name
