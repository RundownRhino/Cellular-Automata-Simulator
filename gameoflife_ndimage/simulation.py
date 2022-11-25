import warnings
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import correlate

from .video import Recorder

# The type to use for neighbour counting. For 2D and neasest neighbours only, a byte should be enough.
COUNTER_TYPE = np.uint8
# The neighbour-counting kernel
neighbour_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(COUNTER_TYPE)


class Rules2D:
    """
    A class representing the ruleset for a 2D cellular automaton.
    """

    def __init__(self, come_to_live: List[int], die: List[int], mode: str = "wrap"):
        """
        :param come_to_live: Neighbour counts of a dead cell that will cause it to come alive.

        :param die: Neighbour counts of a live cell that will cause it to die.

        :param mode: Mode for the neighbour calculation - determines the topology of the zone.
            For example, "wrap"(default) results in cells at the right edge being adjacent to ones
            on the left edge,
            while "constant" results in "cells beyond the edge" to always be considered dead.
            Also see documentation of calculate_neighbours.
        """
        come_to_live = list(come_to_live)
        die = list(die)
        if not all(isinstance(i, int) and 0 <= i <= 8 for i in come_to_live):
            raise ValueError
        if not all(isinstance(i, int) and 0 <= i <= 8 for i in die):
            raise ValueError
        if len(set(come_to_live)) != len(come_to_live):
            raise ValueError("There are duplicates in come_to_live!")
        if len(set(die)) != len(die):
            raise ValueError("There are duplicates in die!")
        # We now know the two lists contain unique (but not
        # necessarity between the two lists) integers from 0 to 8
        self.come_to_live = np.array(sorted(come_to_live), dtype=COUNTER_TYPE)
        self.die = np.array(sorted(die), dtype=COUNTER_TYPE)
        self.mode = mode

    @classmethod
    def classic(cls, mode: str = "wrap") -> "Rules2D":
        """
        :return: The "classic" Convay's Game of Life rules: come_to_live=[3], die = [0,1,4,5,6,7,8]
        """
        return cls(come_to_live=[3], die=[0, 1, 4, 5, 6, 7, 8], mode=mode)


class DrawParams:
    def __init__(self, dead_color: Iterable[int], alive_color: Iterable[int], resize_factor: int = 1):
        """
        :param dead_color: The color to use for dead cells, in RGB format as integers from 0 to 255 inclusive.
        :param alive_color: The color to use for live cells.
        :param resize_factor: The factor to resize the image by when generating.
            Must be at least 1. A resize factor of 1 (default) means each cell will be a pixel, while
            a resize factor of 4 would make each cell drawn as a 4x4 square of 16 pixels.
        """
        self.dead_color = dead_color
        self.alive_color = alive_color
        self.resize_factor = resize_factor
        if not isinstance(self.resize_factor, int):
            raise TypeError
        if self.resize_factor < 1:
            raise ValueError
        alive = np.array(self.alive_color).reshape((3,))
        dead = np.array(self.dead_color).reshape((3,))
        for arr in (alive, dead):
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                warnings.warn(
                    "[DrawParams.__post_init__] Received an array of floating-point color with a max below 1. "
                    "This will be interpreted as black - if you're using floating-point colors from "
                    "0 to 1, rescale them to [0,255]."
                )
        self.alive_color = alive.astype(np.uint8)
        self.dead_color = dead.astype(np.uint8)


@dataclass
class State2D:
    field: np.ndarray
    rules: Rules2D

    @property
    def wh(self) -> Tuple[int, int]:
        """Width and height"""
        return tuple(map(int, self.field.shape[:2][::-1]))  # type:ignore

    def step(self) -> "State2D":
        """
        Calculates the next state. Does not alter self.
        """
        neighbours = calculate_neighbours(self.field, self.rules.mode)
        new_field = self.field.copy()

        # Dead cells:
        notfield = np.logical_not(self.field)
        # Neighbours of alive cells:
        live_neighbours = neighbours[self.field]
        # Neighbours of dead cells:
        dead_neighbours = neighbours[notfield]

        # Live cells with the right neighbour counts die:
        new_field[self.field] = np.logical_not(np.isin(live_neighbours, self.rules.die))

        # Dead cells with the right neighbour counts come to life:
        new_field[notfield] = np.isin(dead_neighbours, self.rules.come_to_live)
        return State2D(field=new_field, rules=self.rules)

    def _to_image_array_noresize(self, draw_params: DrawParams) -> np.ndarray:
        """Utility method to convert to array without considering the resize factor"""
        img_arr = np.full((*self.field.shape, 3), draw_params.dead_color, dtype=np.uint8)
        img_arr[self.field] = draw_params.alive_color
        return img_arr

    def to_image_array(self, draw_params: DrawParams) -> np.ndarray:
        if draw_params.resize_factor == 1:
            return self._to_image_array_noresize(draw_params)
        return np.array(self.to_image(draw_params))  # type:ignore

    def to_image(self, draw_params: DrawParams) -> "Image.Image":
        unresized = self._to_image_array_noresize(draw_params=draw_params)
        img = Image.fromarray(unresized)
        if draw_params.resize_factor == 1:
            return img
        newsize = (img.size[0] * draw_params.resize_factor, img.size[1] * draw_params.resize_factor)
        return img.resize(size=newsize, resample=Image.NEAREST)

    @classmethod
    def random(cls, rules: Rules2D, size: Tuple[int, int]) -> "State2D":
        rng = np.random.default_rng()
        field = rng.integers(0, 2, size=size, dtype=bool)
        return cls(field, rules)

    def run(self, ticks: int) -> Iterable["State2D"]:
        """
        Run the automaton from this state, yielding each state including this one.

        :param ticks: Ticks to simulate. Must be non-negative; zero does nothing and yields the current state.

        :return: Yields a total of ticks+1 states.
        """
        if ticks < 0:
            raise ValueError
        cur_state = self
        yield cur_state
        for tick in range(ticks):
            cur_state = cur_state.step()
            yield cur_state

    def run_and_record(self, ticks: int, draw_params: DrawParams, recorder: Recorder) -> "State2D":
        """
        Run and record each state (including this one, so a total of ticks+1) with the recorder.
        Does not close the recorder afterwards.

        :param ticks: Ticks to simulate. Must be non-negative; zero does nothing and yields the current state.

        :param recorder:

        :param draw_params: The parameters to draw each frame

        :return: The final state
        """
        state = self
        for state in self.run(ticks):
            frame = state.to_image_array(draw_params)
            recorder.send_frame(frame)
        return state


def calculate_neighbours(field: np.ndarray, mode: str = "wrap") -> np.ndarray:
    """
    Returns the number of neighbours for each cell of the array.

    :param field: A n-d ndarray of booleans specifying the state of each cell
    :param mode: Mode to use for the correlation - usually either wrap or constant (with cval=False) depending on the topology.

    :return: A n-d ndarray of the same shape of dtype COUNTER_TYPE of the neighbour counts of each cell.
    """
    # TODO: Technically, not tested for n!=2.
    res = np.zeros(field.shape, dtype=COUNTER_TYPE)
    correlate(field, neighbour_kernel, output=res, mode=mode, cval=False)
    return res
