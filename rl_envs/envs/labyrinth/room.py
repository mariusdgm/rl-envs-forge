from abc import ABC, abstractmethod, abstractclassmethod
import math
import random
import numpy as np
from .constants import WALL, PATH


class RoomFactory:
    def __init__(
        self,
        access_points_nr=None,
        access_points_range=(1, 4),
        room_types=None,
        ratio=None,
        ratio_range=(0.5, 1.5),
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, 1e6)

        self.py_random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)

        self.access_points_nr = access_points_nr
        self.access_points_range = access_points_range

        all_available_room_types = [
            "rectangular",
        ]
        self.room_types = room_types
        if self.room_types is None:
            self.room_types = all_available_room_types

        self.ratio = ratio
        self.ratio_range = ratio_range

    def create_room(
        self, rows=None, cols=None, desired_area=None,
    ):
        if desired_area:
            if desired_area <= 3:
                raise ValueError(
                    f"Attempted to create a room with area {desired_area} but desired area must be greater than 3."
                )
            
            # Setup ratio
            if self.ratio:
                ratio = self.ratio
            else:
                ratio = self.py_random.uniform(*self.ratio_range)
                
            rows, cols = self._estimate_dimensions_from_area(desired_area, ratio)

        # Setup number of access points
        if self.access_points_nr:
            nr_access_points = self.access_points_nr
        else:
            nr_access_points = self.py_random.randint(
                self.access_points_range[0], self.access_points_range[1]
            )
    
        room_type = self.py_random.choice(self.room_types)

        if room_type == "rectangular":
            return self.create_rectangular_room(rows=rows, 
                                                cols=cols, 
                                                nr_access_points=nr_access_points)

    def _estimate_dimensions_from_area(self, desired_area, ratio=1):
        """Estimate room dimensions based on desired area and given ratio."""

        cols = int(math.sqrt(desired_area / ratio))
        rows = int(ratio * cols)

        return rows, cols

    def create_rectangular_room(self, rows=None, cols=None, nr_access_points=1, ):
        """Create a rectangular room using given rows, columns or desired area."""

        return RectangularRoom(rows=rows, cols=cols, nr_access_points=nr_access_points)


class Room(ABC):
    def __init__(
        self,
        rows=None,
        cols=None,
        nr_access_points=1,
        min_rows=3,
        min_cols=3,
        seed=None,
    ):
        """The main way of building a shape will start from the desired area we want the room to have,
        but alternative constructors are possible."""

        self.rows = max(rows, min_rows)
        self.cols = max(cols, min_cols)

        if seed is None:
            seed = random.randint(0, 1e6)

        self.np_random = np.random.RandomState(seed)

        self.top_left_coord = (0, 0)  # Default to the origin
        self.bottom_right_coord = (self.rows, self.cols)
        self.nr_access_points = nr_access_points
        self.access_points = set()
        self.grid = np.ones((self.rows, self.cols), dtype=int) * WALL

    @property
    def area(self):
        return self.rows * self.cols

    @property
    def global_position(self):
        return self.top_left_coord

    @global_position.setter
    def global_position(self, position):
        self.top_left_coord = position

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

        Access points should be represented as a set of (x, y) tuples.
        """
        perimeter_cells = self.get_perimeter_cells()
        chosen_indices = self.np_random.choice(
            len(perimeter_cells), self.nr_access_points, replace=False
        )
        self.access_points = {perimeter_cells[i] for i in chosen_indices}


class RectangularRoom(Room):
    def __init__(self, rows=5, cols=5, nr_access_points=1, min_rows=3, min_cols=3, seed=None):
        super().__init__(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            min_rows=min_rows,
            min_cols=min_cols,
            seed=seed,
        )

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
        return perimeter


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
