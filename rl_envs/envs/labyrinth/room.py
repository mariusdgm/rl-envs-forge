from abc import ABC, abstractmethod, abstractclassmethod
import math
import random
import numpy as np
from .constants import WALL, PATH


class Room(ABC):
    def __init__(self, rows=None, cols=None, nr_access_points=1, **kwargs):
        """The main way of building a shape will start from the desired area we want the room to have,
        but alternative constructors are possible."""

        self.rows = rows
        self.cols = cols
        self.top_left_coord = (0, 0)  # Default to the origin
        self.bottom_right_coord = (rows, cols)
        self.nr_access_points = nr_access_points
        self.grid = np.ones((rows, cols), dtype=int) * WALL

        # If you have specific known attributes, consider unpacking them explicitly here.
        # For instance: self.width = kwargs.get("width", None)
        self.kwargs = kwargs

    @property
    def area(self):
        return self.rows * self.cols

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates

    def get_coordinates(self):
        return self.coordinates

    @property
    def shape(self):
        return self.rows, self.cols

    @abstractmethod
    def generate_room_layout(self):
        pass

    @abstractmethod
    def get_perimeter_cells(self):
        """Get a list of the coordinates of the perimeter cells."""
        pass

    def set_access_points(self):
        """Define default access points for the room, based on its shape.

        Access points should be represented as a list of (x, y) tuples.
        """
        self.access_points = self.get_perimeter_cells()[
            np.random.choice(
                self.get_perimeter_cells().shape[0],
                self.nr_access_points,
                replace=False,
            )
        ]


class RoomFactory:
    def __init__(self, ratio_range=None):
        self.ratio_range = ratio_range

    def create_room(
        self, rows=None, cols=None, desired_area=None, nr_access_points=1, ratio=1
    ):
        if desired_area:
            if desired_area <= 3:
                raise ValueError(
                    f"Attempted to create a room with area {desired_area} but desired area must be greater than 3."
                )
            rows, cols = self._estimate_dimensions_from_area(desired_area, ratio)

        return self.create_rectangular_room(
            rows, cols, nr_access_points=nr_access_points
        )

    def _estimate_dimensions_from_area(self, desired_area, ratio=1):
        """Estimate room dimensions based on desired area and given ratio."""

        if self.ratio_range:
            ratio = random.uniform(*self.ratio_range)
        else:
            ratio = ratio

        cols = int(math.sqrt(desired_area / ratio))
        rows = int(ratio * cols)

        return rows, cols

    def create_rectangular_room(self, rows=None, cols=None, nr_access_points=1):
        """Create a rectangular room using given rows, columns or desired area."""

        # Enforce minimum dimensions specific for this class
        rows = max(rows, 2)
        cols = max(cols, 2)

        return RectangularRoom(rows=rows, cols=cols, nr_access_points=nr_access_points)


class RectangularRoom(Room):
    def __init__(self, rows=10, cols=10, nr_access_points=1, **kwargs):
        super().__init__(rows=rows, cols=cols, **kwargs)
        self.nr_access_points = nr_access_points
        self.generate_room_layout()
        self.set_access_points()

    def generate_room_layout(self):
        self.grid[:] = PATH

    def get_perimeter_cells(self):
        top = [(0, i) for i in range(self.cols)]
        bottom = [(self.rows - 1, i) for i in range(self.cols)]

        # careful of adding corners a second time (start from 1 instead of 0)
        left = [(i, 0) for i in range(1, self.rows - 1)]
        right = [(i, self.cols - 1) for i in range(1, self.rows - 1)]

        perimeter = top + bottom + left + right
        return np.array(perimeter)


class CircularRoom(Room):
    # Implement similar methods for CircularRoom, approximating a circle shape on a grid.
    pass


class DonutRoom(Room):
    # Implement similar methods for DonutRoom, approximating a donut shape on a grid.
    pass


class LShapedRoom(Room):
    # Implement similar methods for LShapedRoom, approximating an L shape on a grid.
    pass


class TShapedRoom(Room):
    pass


class TriangleRoom(Room):
    # Implement similar methods for LShapedRoom, approximating an L shape on a grid.
    pass
