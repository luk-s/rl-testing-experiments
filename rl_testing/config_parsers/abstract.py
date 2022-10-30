import configparser
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

TConfig = TypeVar("TConfig", bound="Config")
T = TypeVar("T")


class Config:
    CONFIG_FOLDER = Path("./configs/")
    DEFAULT_CONFIG_NAME = Path("default.ini")
    REQUIRED_ATTRIBUTES = []
    OPTIONAL_ATTRIBUTES = []

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
        _initialize: bool = False,
    ):

        # Assign the parameters from the provided config file
        if _initialize:
            self.set_parameters(config=config)
            self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        raise NotImplementedError()

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

    @classmethod
    def from_config_file(
        cls: Type[TConfig], config_name: str, _base_config_list: Optional[List[str]] = None
    ) -> TConfig:

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
    def default_config(cls: Type[TConfig]) -> TConfig:
        return cls.from_config_file(cls.DEFAULT_CONFIG_NAME)

    @classmethod
    def set_config_folder_path(cls, path: Union[str, Path]) -> None:
        cls.CONFIG_FOLDER = Path(path)

    @classmethod
    def parse_string(cls, s: str, parser: Optional[Type[T]] = None, raise_error=True) -> T:
        if parser is not None:
            return parser(s)

        if s == "None":
            return None
        elif s == "True":
            return True
        elif s == "False":
            return False
        else:
            try:
                parsed = int(s)
                return parsed
            except ValueError:
                pass
            try:
                parsed = float(s)
                return parsed
            except ValueError:
                if raise_error:
                    raise ValueError(
                        "Input string was not recognized to be any of "
                        f"{[None, True, False, int, float]}. "
                        "Please specify the parser class 'parser'"
                    )
        return s
