import configparser
from pathlib import Path
from typing import Any, Dict, Union

from rl_testing.config_parsers.abstract import Config


class BoardGeneratorConfig(Config):
    CONFIG_FOLDER = Path("./configs/data_generator_configs")
    DEFAULT_CONFIG_NAME = Path("random_default.ini")
    REQUIRED_ATTRIBUTES = ["board_generator_config"]
    OPTIONAL_ATTRIBUTES = []

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
        _initialize: bool = True,
    ):
        # Initialize the parameters
        self.board_generator_config = {}

        # Assign the parameters from the provided config file
        if _initialize:
            self.set_parameters(config=config)
            self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        if section == "DataGeneratorConfig":
            self.board_generator_config[name] = self.parse_string(value, raise_error=False)
        elif (
            name in BoardGeneratorConfig.REQUIRED_ATTRIBUTES
            or name in BoardGeneratorConfig.OPTIONAL_ATTRIBUTES
        ):
            setattr(self, name, self.parse_string(value, raise_error=False))
        else:
            raise ValueError(f"Objects of type {type(self)} don't have the attribute {value}!")


if __name__ == "__main__":
    c = BoardGeneratorConfig.default_config()
    print("finished")
