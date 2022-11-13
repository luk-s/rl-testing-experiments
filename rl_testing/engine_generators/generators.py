import asyncio
from getpass import getpass
from pathlib import Path
from typing import Any, Tuple, Type, TypeVar

import asyncssh
import chess
import chess.engine
from chess.engine import UciProtocol
from rl_testing.config_parsers.engine_config_parser import EngineConfig, RemoteEngineConfig
from rl_testing.engine_generators.relaxed_uci_protocol import RelaxedUciProtocol, popen_uci_relaxed

TUciProtocol = TypeVar("TUciProtocol", bound="UciProtocol")


class EngineGenerator:
    def __init__(self, config: EngineConfig) -> None:
        self.engine_path = config.engine_path
        self.network_base_path = config.network_base_path
        self.engine_config = config.engine_config
        self.network_path = config.network_path

    async def _create_engine(
        self, **kwargs: Any
    ) -> Tuple[asyncio.SubprocessTransport, UciProtocol]:
        return await popen_uci_relaxed(
            self.engine_path,
        )

    async def get_initialized_engine(
        self, initialize_network: bool = True, **kwargs: Any
    ) -> TUciProtocol:
        if initialize_network:
            assert self.network_path is not None, (
                "You first need to set a network using the 'set_network' "
                "function before initializing the engine!"
            )

        # Create the engine
        _, engine = await self._create_engine(**kwargs)

        # Initialize the engine
        if not engine.initialized:
            await engine.initialize()

        # Configure engine
        config = dict(self.engine_config)
        if initialize_network:
            config["WeightsFile"] = self.network_path

        await engine.configure(config)

        return engine

    def set_network_base_path(self, path: str):
        self.network_base_path = str(Path(path))

    def set_network(self, network_name: str) -> None:
        self.network_path = str(Path(self.network_base_path) / Path(network_name))

    async def close(self):
        pass


class RemoteEngineGenerator(EngineGenerator):
    def __init__(self, config: RemoteEngineConfig) -> None:
        super().__init__(config)
        self.remote_host = config.remote_host
        self.remote_user = config.remote_user
        self.password_required = config.password_required
        self.connection = None

    async def _create_engine(self) -> Tuple[asyncio.SubprocessTransport, RelaxedUciProtocol]:
        if self.connection is None:
            # Read in the password from the user
            if self.password_required:
                remote_password = getpass(
                    prompt=f"Please specify the SSH password for the user {self.remote_user}:\n"
                )
            # Start connection
            self.connection = await asyncssh.connect(
                self.remote_host, username=self.remote_user, password=remote_password
            )

            # Delete the password as quickly as possible
            remote_password = None

        return await self.connection.create_subprocess(
            RelaxedUciProtocol,
            self.engine_path,
        )

    async def close(self):
        await self.connection.close()
        self.connection = None
        self.remote_password = None
