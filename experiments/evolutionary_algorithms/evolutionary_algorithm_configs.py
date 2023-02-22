import abc
import configparser
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from rl_testing.config_parsers.evolutionary_algorithm_config_parser import (
    EvolutionaryAlgorithmConfig,
)


class SimpleEvolutionaryAlgorithmConfig(EvolutionaryAlgorithmConfig):
    REQUIRED_ATTRIBUTES = [
        "num_runs_per_config",
        "num_workers",
        "num_generations",
        "population_size",
        "probability_decay",
        "early_stopping",
        "early_stopping_value",
        "mutation_probability",
        "crossover_probability",
    ]
    OPTIONAL_ATTRIBUTES = []

    def __init__(
        self, config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser]
    ) -> None:
        super().__init__(config, _initialize=False)

        # General attributes
        self.num_runs_per_config = None
        self.num_workers = None
        self.num_generations = None
        self.population_size = None
        self.probability_decay = None
        self.early_stopping = None
        self.early_stopping_value = None
        self.mutation_probability = None
        self.crossover_probability = None

        self.set_parameters(config=config)
        self.check_parameters()

    def set_parameter(self, section: str, name: str, value: Any) -> None:
        # Parse the value
        if isinstance(value, str):
            value = self.parse_string(value, raise_error=False)

        # Check if the value belongs to this class or the parent class
        if name in self.REQUIRED_ATTRIBUTES or name in self.OPTIONAL_ATTRIBUTES:
            setattr(self, name, value)
        else:
            super().set_parameter(section, name, value)

    def check_parameters(self) -> None:
        super().check_parameters()

        # Check that probability decay is only enabled if mutation_strategy is "all" and if
        # crossover_strategy is "all"
        if self.probability_decay:
            if self.mutator_attributes["mutation_strategy"] != "all":
                raise ValueError(
                    "Probability decay can only be enabled if mutation_strategy is 'all'"
                )
            if self.crossover_attributes["crossover_strategy"] != "all":
                raise ValueError(
                    "Probability decay can only be enabled if crossover_strategy is 'all'"
                )
