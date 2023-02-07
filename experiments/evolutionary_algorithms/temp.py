import random
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from chess.engine import Cp, Mate, Score
from deap import base, creator, tools

from rl_testing.util.util import cp2q


class Noise2DPiecewise:
    """A smooth, continuous 2d noise function.
    Uses the idea from the video "Painting a Landscape with Maths", by  Inigo Quilez
    see here: https://www.youtube.com/watch?v=BFld4EBO2RE (starting at 02:00 minutes)
    """

    def __init__(
        self,
        min_corner: Iterable,
        max_corner: Iterable,
        num_rows: int,
        num_cols: int,
        min_val: int,
        max_val: int,
        seed: Optional[int] = None,
    ):
        """Instantiate the noise class

        Args:
            min_corner (Iterable): The smallest corner of the box environment.
            max_corner (Iterable): The largest corner of the box environment.
            num_rows (int): The number of rows in the noise grid
            num_cols (int): The number of columns in the noise grid
            min_val (int): The minimum noise value
            max_val (int): The maximum noise value
            seed (Optional[int], optional): A seed for the random number generator.
                Defaults to None.

        Raises:
            ValueError: If 'num_rows' or 'num_cols' is not a positive integer
            ValueError: If 'min_val' is larger than 'max_val'
        """
        if seed is not None:
            np.random.seed(seed)

        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)

        if not (self.min_corner <= self.max_corner).all():
            raise ValueError("'min_corner' must be <= than 'max_corner'")

        self.x_min, self.y_min = self.min_corner
        self.x_max, self.y_max = self.max_corner

        self.num_rows = np.array(num_rows)
        self.num_cols = np.array(num_cols)
        self.min_val = np.array(min_val)
        self.max_val = np.array(max_val)

        if (
            not np.issubdtype(self.num_rows.dtype, np.integer)
            or not self.num_rows > 0
            or not np.issubdtype(self.num_cols.dtype, np.integer)
            or not self.num_cols > 0
        ):
            raise ValueError(
                "'num_rows' and 'num_cols' need to be positive integers, but got "
                f"{num_rows=}, {num_cols=}."
            )

        if not self.min_val <= self.max_val:
            raise ValueError(
                "'min_val' must be <= than 'max_val' but got " f"{self.min_val=}, {self.max_val=}"
            )

        # Get the dimensions of a single grid cell
        self.cell_dims = (self.max_corner - self.min_corner) / (
            self.num_rows,
            self.num_cols,
        )

        # Build the grid of corner vertices
        self.grid_values = np.random.uniform(
            self.min_val, self.max_val, size=(self.num_rows + 1, self.num_cols + 1)
        )

    def __call__(self, pos: Iterable) -> np.ndarray:
        """Takes in an array of positions in the environment and returns the noise value at each
        position. Basically just performs a few safety checks and then calls the _get method.

        Args:
            pos (Iterable): An array of positions in the environment

        Raises:
            ValueError: If pos is not a numpy 1d/2d array
            ValueError: If the last dimension of pos is not 2
            ValueError: If not all values inside pos are inside the environment

        Returns:
            np.ndarray: An array of noise values
        """
        pos_np = np.array(pos)
        if len(pos_np.shape) > 2:
            raise ValueError("'pos' needs to be either a 1d or 2d array.")
        if pos_np.shape[-1] != 2:
            raise ValueError(
                f"The last dimension of 'pos' must be 2 but got {pos_np.shape[-1]} instead"
            )
        # Check that pos is inside the 2d range
        if not (self.min_corner <= pos_np).all() or not (pos_np <= self.max_corner).all():
            raise ValueError(
                "All values inside 'pos' must be >= than "
                f"{self.min_corner} and <= than {self.max_corner}"
            )

        # Make sure that 'pos' is a 2d array
        if len(pos_np.shape) == 1:
            pos_np = pos_np[np.newaxis, ...]

        return self._get(pos_np)

    def _smooth_step(self, x: np.ndarray, a: int = 0, b: int = 1) -> np.ndarray:
        """Computes the noise at position x, y in a grid cell of size a, b.

        Args:
            x (np.ndarray): The position in the grid cell
            a (int, optional): The width of the grid cell. Defaults to 0.
            b (int, optional): The height of the grid cell. Defaults to 1.

        Returns:
            np.ndarray: The noise value at position x, y
        """
        theta = np.minimum(1, np.maximum(0, (x - a) / (b - a)))
        return 3 * (theta**2) - 2 * (theta**3)

    def _get(self, pos: np.ndarray) -> np.ndarray:
        """Takes in an array of positions in the environment and returns the noise value at each
        position.

        Args:
            pos (np.ndarray): An array of positions in the environment

        Returns:
            np.ndarray: An array of noise values
        """
        x_dim, y_dim = self.cell_dims

        # Compute for each tuple in pos the 4 corners of the tile it is contained in
        a_pos = ((pos - self.min_corner) // self.cell_dims).astype(np.int32)

        # a-corners can't take the maximum value. => fix that
        first_maximum = np.where(a_pos[:, 0] == len(self.grid_values) - 1)
        second_maximum = np.where(a_pos[:, 1] == len(self.grid_values[0]) - 1)
        a_pos[first_maximum] -= (1, 0)
        a_pos[second_maximum] -= (0, 1)

        # Compute positions of other vertices
        b_pos = (a_pos + (1, 0)).astype(np.int32)
        c_pos = (a_pos + (0, 1)).astype(np.int32)
        d_pos = (a_pos + (1, 1)).astype(np.int32)

        # Extract the corresponding heights of each corner
        a = self.grid_values[a_pos[:, 0], a_pos[:, 1]]
        b = self.grid_values[b_pos[:, 0], b_pos[:, 1]]
        c = self.grid_values[c_pos[:, 0], c_pos[:, 1]]
        d = self.grid_values[d_pos[:, 0], d_pos[:, 1]]

        # Compute the two smooth steps
        x_smooth_step = self._smooth_step(
            pos[:, 0] - self.min_corner[0] - a_pos[:, 0] * x_dim, 0, x_dim
        )
        y_smooth_step = self._smooth_step(
            pos[:, 1] - self.min_corner[1] - a_pos[:, 1] * y_dim, 0, y_dim
        )

        # Compute the final noise values
        noise_values = (
            a
            + (b - a) * x_smooth_step
            + (c - a) * y_smooth_step
            + (a - b - c + d) * x_smooth_step * y_smooth_step
        )

        return noise_values


def main():
    pop = toolbox.population(n=50)
    CROSSOVER_PROB, MUTATION_PROB, N_GENERATIONS = 0.5, 0.2, 13

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(N_GENERATIONS):
        print(f"{g = }")
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop


if __name__ == "__main__":
    """
    ################
    # CONFIG START #
    ################
    seed = 42

    # image resolution
    resolution = 300

    # Number of rows and columns
    num_rows = 10
    num_cols = 10

    # Min and max values
    min_val = -2
    max_val = 2

    # Grid min and max corners
    min_corner = np.array([0, 0])
    max_corner = np.array([10, 10])
    ################
    #  CONFIG END  #
    ################

    # Set the random seeds
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Create the noise generator
    noise2d = Noise2DPiecewise(
        seed=seed,
        min_corner=min_corner,
        max_corner=max_corner,
        num_rows=10,
        num_cols=10,
        min_val=min_val,
        max_val=max_val,
    )

    # Create a 2d grid of points
    x = np.meshgrid(*[np.linspace(min_corner[i], max_corner[i], resolution) for i in range(2)])
    x2 = np.vstack([x[0].flatten(), x[1].flatten()]).T

    # Compute the noise values at each point
    noise_values = noise2d(x2)

    # Plot the noise values
    # plt.imshow(noise_values.reshape(resolution, resolution), origin="lower")
    plt.contourf(x[0], x[1], noise_values.reshape(resolution, resolution), 20)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    IND_SIZE = 2

    toolbox = base.Toolbox()
    toolbox.register("attribute", lambda: np.random.uniform(min_corner[0], max_corner[0]))
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return (noise2d(individual),)

    def feasible(individual):
        return (min_corner <= individual).all() and (max_corner >= individual).all()

    def distance(individual):
        return (-min(min(individual - min_corner), 0) + max(max(individual - max_corner), 0)) ** 2

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, -10, distance))

    best_individuals = np.array(main()).T
    print("Best individuals: ", best_individuals.T)

    plt.scatter(best_individuals[0], best_individuals[1], c="red", s=60, marker="x")

    plt.show()
    """
    import chess

    # board = chess.Board("3B2b1/P7/r5Rr/1ppB2PP/r6b/2pP1ppP/p1qppP2/1QkNNB1K b - - 3 43")
    # board = chess.Board("8/pKPPP1P1/Bb1pP2R/NpQ3bp/1pPr3p/1P1q3B/1n1N1pPp/2k5 w - - 20 187")
    board = chess.Board("1r6/pN4K1/3R4/n1R2qp1/PpPp2qB/6p1/p1kPpp2/bB5r b - - 46 117")
    board = chess.Board("1r6/3P2K1/1P1R2P1/1PR2qp1/N2P1NqB/2p1P3/p1k2ppn/bB6 b - - 45 117")
    board = chess.Board("8/r2P2K1/1P1R2P1/1P3pp1/N2P1NqB/2p1P3/p1k2ppn/bB6 b - - 45 117")
    board = chess.Board("4b3/1N3np1/1rNp4/pb1kp2P/2R3KB/Q1pp4/P2nP3/1q2r3 w - - 56 5")
    print(board)
    print(board.is_valid())
    print("Computed difference: ", 1.9122740696762588)
    print("Real difference: ", 0.75694 + 0.95296)

    s = Mate(4)
    print("finished")
