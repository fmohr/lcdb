from ConfigSpace import ConfigurationSpace, Float, Uniform
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    IntegerHyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import ForbiddenValueError, deactivate_inactive_hyperparameters

with open("configspace_mlp.json", "r") as f:
    json_string = f.read()
    cs = cs_json.read(json_string)

import warnings
import numpy as np
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import Sobol


# https://github.com/automl/SMAC3/blob/e64e1918eeb88e93f9f201ece343624fb2943e9d/smac/initial_design/sobol_design.py#L17
class SobolSequenceGenerator:
    def __init__(
        self,
        configuration_space: ConfigurationSpace,
        n: int,
        scramble: bool = True,
        seed: int = 0,
    ):
        """
        Generator for Sobol sequences of configurations based on a ConfigurationSpace.

        Parameters
        ----------
        configuration_space : ConfigurationSpace
            The ConfigurationSpace representing the search space for configurations.
        n : int
            The number of configurations to generate.
        scramble : bool, optional (default=True)
            Whether to apply scrambling to the Sobol sequence.
        seed : int, optional (default=0)
            The seed value used for generating the Sobol sequence.

        Raises
        ------
        ValueError
            If the number of hyperparameters in the ConfigurationSpace exceeds 21201.
        """
        self.configuration_space = configuration_space
        self.n = n
        self.scramble = scramble
        self.seed = seed

        if len(self.configuration_space.get_hyperparameters()) > 21201:
            raise ValueError(
                "The default Sobol sequence generator can only handle up to 21201 dimensions."
            )

    def generate(self) -> list[Configuration]:
        """
        Generate the Sobol sequence of configurations.

        Returns
        -------
        list[Configuration]
            A list of generated configurations based on the Sobol sequence.
        """
        params = self.configuration_space.get_hyperparameters()

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        dim = len(params) - constants
        sobol_gen = Sobol(d=dim, scramble=self.scramble, seed=self.seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = sobol_gen.random(self.n)

        return self._transform_continuous_designs(
            design=sobol,
            origin="Sobol sequence",
            configuration_space=self.configuration_space,
        )

    # https://github.com/automl/SMAC3/blob/e64e1918eeb88e93f9f201ece343624fb2943e9d/smac/initial_design/abstract_initial_design.py#L159
    def _transform_continuous_designs(
        self, design: np.ndarray, origin: str, configuration_space: ConfigurationSpace
    ) -> list[Configuration]:
        """
        Transform the continuous designs obtained from the Sobol sequence into a list of Configuration objects.

        Parameters
        ----------
        design : np.ndarray
            Array of continuous values obtained from the Sobol sequence.
        origin : str
            Label indicating the origin of the configurations.
        config_space : ConfigurationSpace
            ConfigurationSpace object representing the search space.

        Returns
        -------
        configs : list[Configuration]
            List of Configuration objects with the transformed designs.

        Raises
        ------
        ValueError
            If a hyperparameter type other than IntegerHyperparameter, NumericalHyperparameter, Constant,
            CategoricalHyperparameter or OrdinalHyperparameter is encountered.
        """
        params = configuration_space.get_hyperparameters()
        for idx, param in enumerate(params):
            if isinstance(param, IntegerHyperparameter):
                design[:, idx] = param._inverse_transform(
                    param._transform(design[:, idx])
                )
            elif isinstance(param, NumericalHyperparameter):
                continue
            elif isinstance(param, Constant):
                design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
                design_[:, :idx] = design[:, :idx]
                design_[:, idx + 1 :] = design[:, idx:]
                design = design_
            elif isinstance(param, CategoricalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.choices), dtype=int)
            elif isinstance(param, OrdinalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.sequence), dtype=int)
            else:
                raise ValueError(
                    "Hyperparameter not supported when transforming a continuous design."
                )

        configs = []
        for vector in design:
            try:
                conf = deactivate_inactive_hyperparameters(
                    configuration=None,
                    configuration_space=configuration_space,
                    vector=vector,
                )
            except ForbiddenValueError:
                continue

            conf.origin = origin
            configs.append(conf)

        return configs


# Generate a Sobol sequence of size 10
sobol_generator = SobolSequenceGenerator(cs, n=10)
sobol_sequence = sobol_generator.generate()

# Compare to random
random_sequence = cs.sample_configuration(10)

# Test 2D numeric
cs_test = ConfigurationSpace(
    name="test",
    seed=1234,
    space={
        "x1": Float(
            "x1",
            bounds=(-10, 10),
            distribution=Uniform(),
        ),
        "x2": Float(
            "x2",
            bounds=(-10, 10),
            distribution=Uniform(),
        ),
    },
)

# Generate a Sobol sequence of size 16
sobol_generator = SobolSequenceGenerator(cs_test, n=16)
sobol_sequence = sobol_generator.generate()

# Compare to random
random_sequence = cs_test.sample_configuration(16)

# Extract x and y values from the Sobol sequence and random sequence
sobol_x = [config.get("x1") for config in sobol_sequence]
sobol_y = [config.get("x2") for config in sobol_sequence]
random_x = [config.get("x1") for config in random_sequence]
random_y = [config.get("x2") for config in random_sequence]

import matplotlib.pyplot as plt

# Plot the Sobol sequence and random sequence
plt.scatter(sobol_x, sobol_y, c="blue", label="Sobol Sequence")
plt.scatter(random_x, random_y, c="red", label="Random Sequence")
plt.xlabel("x1")
plt.ylabel("x2")
grid_lines = [-10, -5, 0, 5, 10]
plt.hlines(grid_lines, -10, 10, colors="gray", linestyles="dotted")
plt.vlines(grid_lines, -10, 10, colors="gray", linestyles="dotted")
plt.legend()
plt.show()
