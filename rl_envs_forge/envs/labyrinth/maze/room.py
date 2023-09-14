from abc import ABC, abstractmethod, abstractclassmethod
from typing import List, Tuple, Union, Optional
import math
import random
import numpy as np
from scipy.ndimage import binary_dilation

from ..constants import WALL, PATH
from .utils import is_perimeter, generate_ellipse, clamp


class RoomFactory:
    def __init__(
        self,
        access_points_nr: Optional[int] = None,
        access_points_range: Tuple[int, int] = (1, 4),
        room_types: Optional[List[str]] = None,
        ratio: Optional[Union[int, float]] = None,
        ratio_range: Tuple[Union[int, float], Union[int, float]] = (0.5, 1.5),
        seed: Optional[int] = None,
    ) -> None:
        """Construct a RoomFactory object for generating room layouts.

        Arguments `access_points_nr`, `ratio` are used to fix specific values,
        otherwise the values are drawn from the minimum and maximum distributions.

        Argument `room_types` is used to determine what types of rooms are to be added.
        If `None`, the random selection considers all the implemented room types.

        Args:
            access_points_nr (int, optional): The number of access points in the room. Defaults to None.
            access_points_range (tuple, optional): The range of access points per room. Defaults to (1, 4).
            room_types (list, optional): The types of rooms to be added. Defaults to None.
            ratio (float, optional): The room ratio. Defaults to None.
            ratio_range (tuple, optional): The range of room ratio. Defaults to (0.5, 1.5).
            seed (int, optional): The seed to use for generating random numbers. Defaults to None.
        """
        self.seed = seed
        if self.seed is None:
            self.seed = random.randint(0, 1e6)
        self.room_seed = self.seed

        self.py_random = random.Random(self.seed)
        self.np_random = np.random.RandomState(self.seed)

        self.access_points_nr = access_points_nr
        self.access_points_range = access_points_range

        all_available_room_types = [
            "rectangular",
            "oval",
            "donut",
        ]
        self.room_types = room_types
        if self.room_types is None:
            self.room_types = all_available_room_types

        self.ratio = ratio
        self.ratio_range = ratio_range

    def create_room(
        self,
        rows=None,
        cols=None,
        desired_area=None,
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

        # increment room seed to seed each room differently
        self.room_seed += 1

        room_type = self.py_random.choice(self.room_types)

        if room_type == "rectangular":
            return self.create_rectangular_room(
                rows=rows, cols=cols, nr_access_points=nr_access_points
            )
        elif room_type == "oval":
            return self.create_oval_room(
                rows=rows, cols=cols, nr_access_points=nr_access_points
            )
        elif room_type == "donut":
            return self.create_donut_room(
                rows=rows, cols=cols, nr_access_points=nr_access_points
            )

    def _estimate_dimensions_from_area(self, desired_area, ratio=1):
        """Estimate room dimensions based on desired area and given ratio."""

        cols = int(math.sqrt(desired_area / ratio))
        rows = int(ratio * cols)

        return rows, cols

    def create_rectangular_room(self, rows=None, cols=None, nr_access_points=1):
        """Create a rectangular room using given rows, columns or desired area."""
        return RectangularRoom(
            rows=rows, cols=cols, nr_access_points=nr_access_points, seed=self.room_seed
        )

    def create_oval_room(self, rows=None, cols=None, nr_access_points=1):
        """Create an oval room using given rows, columns or desired area."""
        return OvalRoom(
            rows=rows, cols=cols, nr_access_points=nr_access_points, seed=self.room_seed
        )

    def create_donut_room(self, rows=None, cols=None, nr_access_points=1):
        """Create a donut room using given rows, columns or desired area."""
        min_ring_width = 2
        max_ring_width = min(rows, cols) // 3
        if max_ring_width <= min_ring_width:
            ring_width = min_ring_width
        else:
            ring_width = self.py_random.randint(min_ring_width, max_ring_width)
        
        return DonutRoom(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            ring_width=ring_width,
            seed=self.room_seed,
        )


class Room(ABC):
    def __init__(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        nr_access_points: int = 1,
        min_rows: int = 3,
        min_cols: int = 3,
        seed: Optional[int] = None,
        top_left_coord: Tuple[int, int] = (0, 0),
    ) -> None:
        """
        Base Room class.

        Args:
            rows (int, optional): The number of rows in the room. Defaults to None.
            cols (int, optional): The number of columns in the room. Defaults to None.
            nr_access_points (int, optional): The number of access points for the room. Defaults to 1.
            min_rows (int, optional): Minimum rows for the room. Defaults to 3.
            min_cols (int, optional): Minimum columns for the room. Defaults to 3.
            seed (int, optional): Seed value for random operations. Defaults to None.
            top_left_coord (tuple, optional): The top left coordinate of the room. Defaults to (0, 0).
        """

        self.rows = max(rows, min_rows)
        self.cols = max(cols, min_cols)

        self.seed = seed
        if self.seed is None:
            self.seed = random.randint(0, 1e6)

        self.np_random = np.random.RandomState(self.seed)

        self.top_left_coord = top_left_coord
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
    def get_perimeter_cells(self, padding=0):
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

    @abstractmethod
    def generate_inner_area_mask(self, total_rows, total_cols):
        pass


class RectangularRoom(Room):
    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        nr_access_points: int = 1,
        min_rows: int = 3,
        min_cols: int = 3,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            min_rows=min_rows,
            min_cols=min_cols,
            seed=seed,
            **kwargs,
        )
        """Construct a RectangularRoom object representing a rectangular room.

        Args:
            rows (int, optional): The number of rows in the room. Defaults to 5.
            cols (int, optional): The number of columns in the room. Defaults to 5.
            nr_access_points (int, optional): The number of access points in the room. Defaults to 1.
            min_rows (int, optional): The minimum number of rows for the room. Defaults to 3.
            min_cols (int, optional): The minimum number of columns for the room. Defaults to 3.
            seed (int, optional): The seed to use for generating random numbers. Defaults to None.
        """

        self.generate_room_layout()
        self.set_access_points()

    def generate_room_layout(self):
        self.grid[:] = PATH

    def get_perimeter_cells(self, padding=0):
        # Top perimeter with padding
        top = [(-padding, j) for j in range(-padding, self.cols + padding)]

        # Bottom perimeter with padding
        bottom = [
            (self.rows - 1 + padding, j) for j in range(-padding, self.cols + padding)
        ]

        # Left perimeter with padding (excluding the corners)
        left = [(i, -padding) for i in range(1 - padding, self.rows - 1 + padding)]

        # Right perimeter with padding (excluding the corners)
        right = [
            (i, self.cols - 1 + padding)
            for i in range(1 - padding, self.rows - 1 + padding)
        ]

        perimeter = top + bottom + left + right
        return perimeter

    def generate_inner_area_mask(self):
        return np.ones((self.rows, self.cols), dtype=bool)


class OvalRoom(Room):
    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        nr_access_points: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Round down to odd number to have symmetry
        rows = rows - (rows % 2 == 0)
        cols = cols - (cols % 2 == 0)

        super().__init__(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            min_rows=5,
            min_cols=5,
            seed=seed,
            **kwargs,
        )

        self.generate_room_layout()
        self.set_access_points()

    def generate_room_layout(self):
        center_y, center_x = self.rows // 2, self.cols // 2
        a, b = center_x, center_y

        return generate_ellipse(self.rows, self.cols, a, b, center_y, center_x)

    def get_perimeter_cells(self, padding=0):
        room_mask = self.grid == PATH

        # For padding 0, directly return perimeter cells of the room
        if padding == 0:
            perimeter_cells = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if room_mask[i, j] and is_perimeter(room_mask, (i, j)):
                        perimeter_cells.append((i, j))
            return perimeter_cells

        # Expand the mask to include padding on all sides
        expanded_mask = np.pad(
            room_mask, padding, mode="constant", constant_values=False
        )

        # Now, apply dilation to the expanded mask
        dilated_mask = binary_dilation(expanded_mask, iterations=padding)

        # If padding is greater than 1, generate the inner dilated mask
        if padding > 1:
            inner_dilated_mask = binary_dilation(expanded_mask, iterations=padding - 1)
        else:
            inner_dilated_mask = expanded_mask

        perimeter_cells = []
        for i in range(dilated_mask.shape[0]):
            for j in range(dilated_mask.shape[1]):
                if dilated_mask[i, j] and not inner_dilated_mask[i, j]:
                    perimeter_cells.append((i - padding, j - padding))

        return perimeter_cells

    def generate_inner_area_mask(self):
        return (self.grid == PATH).astype(int)


class DonutRoom(Room):
    def __init__(
        self,
        rows: int = 7,
        cols: int = 7,
        nr_access_points: int = 1,
        ring_width: int = 2,
        outer_shape: str = "oval",  # new parameter
        inner_shape: str = "oval",  # new parameter
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Validation for shape
        valid_shapes = ["oval", "rectangular"]
        assert outer_shape in valid_shapes, f"Invalid outer_shape. Expected one of {valid_shapes}"
        assert inner_shape in valid_shapes, f"Invalid inner_shape. Expected one of {valid_shapes}"

        # Round down to odd number to have symmetry
        rows = rows - (rows % 2 == 0)
        cols = cols - (cols % 2 == 0)
        self.ring_width = clamp(ring_width, 2, min(rows, cols)//3)
        self.outer_shape = outer_shape
        self.inner_shape = inner_shape

        super().__init__(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            min_rows=7,
            min_cols=7,
            seed=seed,
            **kwargs,
        )

        self.generate_room_layout()
        self.set_access_points()

    def generate_room_layout(self):
        center_y, center_x = self.rows // 2, self.cols // 2
        a_outer, b_outer = center_x, center_y
        a_inner = a_outer - self.ring_width
        b_inner = b_outer - self.ring_width

        # Depending on the shape, generate the outer and inner masks
        outer_mask = self._generate_shape_mask(self.outer_shape, a_outer, b_outer, center_y, center_x)
        inner_mask = self._generate_shape_mask(self.inner_shape, a_inner, b_inner, center_y, center_x)

        self.grid = outer_mask - inner_mask
        return self.grid

    def _generate_shape_mask(self, shape, a, b, center_y, center_x):
        if shape == "oval":
            return generate_ellipse(self.rows, self.cols, a, b, center_y, center_x)
        elif shape == "rectangular":
            mask = np.zeros((self.rows, self.cols), dtype=int)
            mask[center_y - b:center_y + b, center_x - a:center_x + a] = 1
            return mask

    def generate_inner_area_mask(self):
        if self.outer_shape == "oval":
            center_y, center_x = self.rows // 2, self.cols // 2
            a, b = center_x, center_y
            return generate_ellipse(self.rows, self.cols, a, b, center_x, center_y, value_true=1, value_false=0)
            
        elif self.outer_shape == "rectangular":
            mask = np.ones((self.rows, self.cols), dtype=int)
            return mask


class LShapedRoom(Room):
    # Implement similar methods for LShapedRoom, approximating an L shape on a grid.
    pass


class TShapedRoom(Room):
    pass


class TriangleRoom(Room):
    # Implement similar methods for LShapedRoom, approximating an L shape on a grid.
    pass
