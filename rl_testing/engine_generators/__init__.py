from typing import Union

from rl_testing.config_parsers import (
    LeelaEngineConfig,
    LeelaRemoteEngineConfig,
    StockfishEngineConfig,
    StockfishRemoteEngineConfig,
)
from rl_testing.engine_generators.generators import (
    EngineGenerator,
    RemoteEngineGenerator,
)


def get_engine_generator(
    config: Union[
        LeelaEngineConfig,
        LeelaRemoteEngineConfig,
        StockfishEngineConfig,
        StockfishRemoteEngineConfig,
    ]
) -> Union[EngineGenerator, RemoteEngineGenerator]:
    if type(config) == LeelaEngineConfig or type(config) == StockfishEngineConfig:
        return EngineGenerator(config)
    elif type(config) == LeelaRemoteEngineConfig or type(config) == StockfishRemoteEngineConfig:
        return RemoteEngineGenerator(config)
    else:
        raise ValueError(f"Engine config of type {type(config)} is not supported!")
