import configparser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

EngineConfigs = Union["EngineConfig", "RemoteEngineConfig"]


class EngineConfig:
    CONFIG_FOLDER = Path("./configs/engine_configs")
    DEFAULT_CONFIG_NAME = Path("default.ini")
    REQUIRED_ATTRIBUTES = ["engine_path", "network_base_path", "engine_config", "search_limits"]
    OPTIONAL_ATTRIBUTES = ["network_path"]

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
        _initialize: bool = True,
    ):
        # Initialize the parameters
        self.engine_path = None
        self.network_base_path = None
        self.network_path = None
        self.engine_config = {}
        self.search_limits = {}

        # Assign the parameters from the provided config file
        if _initialize:
            self.set_parameters(config=config)
            self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        if section == "EngineConfig":
            self.engine_config[name] = value
        elif section == "SearchLimits":
            self.search_limits[name] = value
        elif name in EngineConfig.REQUIRED_ATTRIBUTES or name in EngineConfig.OPTIONAL_ATTRIBUTES:
            setattr(self, name, value)
        else:
            raise ValueError(f"Objects of type {type(self)} don't have the attribute {value}!")

    def set_parameters(
        self, config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser]
    ) -> None:
        if isinstance(config, dict):
            parsed = configparser.ConfigParser().read_dict(config)
        elif isinstance(config, configparser.ConfigParser):
            parsed = config
        else:
            raise ValueError(
                "'config' must be either of type 'dict' or type 'configparser.ConfigParser'"
            )

        for section in parsed.sections():
            for name, value in parsed.items(section):
                self.set_parameter(section, name, value)

    def check_parameters(self) -> None:
        for name in self.REQUIRED_ATTRIBUTES:
            if not hasattr(self, name) or getattr(self, name) is None:
                raise ValueError(f"Parameter '{name}' must be defined!")

    def set_network(self, network_name: str) -> None:
        self.network_path = str(Path(self.network_base_path) / Path(network_name))

    @classmethod
    def from_config_file(
        cls, config_name: str, _base_config_list: Optional[List[str]] = None
    ) -> Union["EngineConfig", "RemoteEngineConfig"]:

        if _base_config_list is None:
            _base_config_list = [config_name]

        # Create the config parser
        config_parser = configparser.ConfigParser()
        file_path = str(cls.CONFIG_FOLDER / Path(config_name))
        with open(file_path, "r") as f:
            config_parser.read_file(f)

        # Check if the config is based on another root config
        if "BaseConfig" in config_parser.sections():
            # Load the root config file
            base_config_name = Path(config_parser["BaseConfig"]["base_config_name"])

            # Make sure that no circular import is created
            if base_config_name in _base_config_list:
                raise ValueError(f"Circular import detected! {_base_config_list}")

            _base_config_list.append(base_config_name)

            config_file = cls.from_config_file(
                base_config_name, _base_config_list=_base_config_list
            )

            # Remove the 'BaseConfig' section
            config_parser.remove_section("BaseConfig")

            # Update the config with the new parameters
            config_file.set_parameters(config_parser)

            return config_file
        else:
            # If the config is not based on another config, return this config
            return cls(config_parser)

    @classmethod
    def default_config(cls) -> Union["EngineConfig", "RemoteEngineConfig"]:
        return cls.from_config_file(cls.DEFAULT_CONFIG_NAME)

    @classmethod
    def set_config_folder_path(cls, path: Union[str, Path]) -> None:
        cls.CONFIG_FOLDER = Path(path)


class RemoteEngineConfig(EngineConfig):
    DEFAULT_CONFIG_NAME = Path("default_remote.ini")
    REQUIRED_ATTRIBUTES = ["remote_host", "remote_user", "password_required"]
    OPTIONAL_ATTRIBUTES = ["network_path"]

    def __init__(self, config: Union[Dict[str, Any], configparser.ConfigParser]):
        super().__init__(config=config, _initialize=False)

        self.remote_host = None
        self.remote_user = None
        self.password_required = None

        self.set_parameters(config=config)
        self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        if name == "password_required":
            self.password_required = value == "True"
        elif (
            name in RemoteEngineConfig.REQUIRED_ATTRIBUTES
            or name in RemoteEngineConfig.OPTIONAL_ATTRIBUTES
        ):
            setattr(self, name, value)
        else:
            super().set_parameter(section, name, value)

    def check_parameters(self) -> None:
        for name in self.REQUIRED_ATTRIBUTES:
            if not hasattr(self, name) or getattr(self, name) is None:
                raise ValueError(f"Parameter '{name}' must be defined!")
        super().check_parameters()


if __name__ == "__main__":
    r = RemoteEngineConfig.default_config()
    s = RemoteEngineConfig.from_config_file("remote_400_nodes.ini")
    print("finished")
