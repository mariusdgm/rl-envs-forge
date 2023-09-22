from abc import ABC, abstractmethod, abstractclassmethod
from typing import List, Tuple, Union, Optional
import math
import random
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from ..constants import WALL, PATH
from .utils import generate_ellipse, clamp


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

        self.all_available_room_types = [
            "rectangle",
            "oval",
            "donut",
            "t_shape",
            "l_shape",
            "triangle",
        ]
        self.room_types = room_types
        if self.room_types is None:
            self.room_types = self.all_available_room_types

        self.ratio = ratio
        self.ratio_range = ratio_range

    def create_room(
        self,
        rows: int = None,
        cols: int = None,
        desired_area: float = None,
    ) -> "Room":
        """
        Create a room with the specified dimensions or desired area.

        Parameters:
            rows (int): The number of rows in the room. Default is None.
            cols (int): The number of columns in the room. Default is None.
            desired_area (float): The desired area of the room. Default is None.

        Returns:
            Room: The created room object.

        Raises:
            ValueError: If the desired area is less than or equal to 3.

        Notes:
            - If desired_area is provided, it takes precedence over rows and cols.
            - If ratio is not set, it is randomly selected from the ratio_range.
            - The number of access points is either specified by access_points_nr or randomly selected from the access_points_range.
            - Each room has a unique seed value.
            - The type of the room is randomly selected from the available room types.

        Available room types:
            - rectangle
            - oval
            - donut
            - t_shape
            - l_shape
            - triangle
        """
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

        if room_type == "rectangle":
            return self.create_rectangle_room(
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
        elif room_type == "t_shape":
            return self.create_t_shape_room(
                rows=rows, cols=cols, nr_access_points=nr_access_points
            )
        elif room_type == "l_shape":
            return self.create_l_shape_room(
                rows=rows, cols=cols, nr_access_points=nr_access_points
            )
        elif room_type == "triangle":
            return self.create_triangle_room(
                rows=rows, cols=cols, nr_access_points=nr_access_points
            )
        else:
            raise ValueError(
                f"Unknown room type: {room_type}. Available types: {self.all_available_room_types}"
            )

    def _estimate_dimensions_from_area(
        self, desired_area: float, ratio: float = 1
    ) -> Tuple[int, int]:
        """Estimate room dimensions based on desired area and given ratio."""

        cols = int(math.sqrt(desired_area / ratio))
        rows = int(ratio * cols)

        return rows, cols

    def create_rectangle_room(
        self, rows: int = None, cols: int = None, nr_access_points: int = 1
    ) -> "RectangleRoom":
        """
        Create a new rectangle room with the specified number of rows, columns, and
        number of access points.

        Args:
            rows (int): The number of rows in the room. If not specified, a default value
                will be used.
            cols (int): The number of columns in the room. If not specified, a default
                value will be used.
            nr_access_points (int): The number of access points in the room. If not
                specified, a default value will be used.

        Returns:
            RectangleRoom: The newly created rectangle room.
        """

        return RectangleRoom(
            rows=rows, cols=cols, nr_access_points=nr_access_points, seed=self.room_seed
        )

    def create_oval_room(
        self, rows: int = None, cols: int = None, nr_access_points: int = 1
    ) -> "OvalRoom":
        """
        Creates an `OvalRoom` object with the specified dimensions and number of access points.

        Parameters:
            rows (int, optional): The number of rows in the room. Defaults to None.
            cols (int, optional): The number of columns in the room. Defaults to None.
            nr_access_points (int, optional): The number of access points in the room. Defaults to 1.

        Returns:
            OvalRoom: An `OvalRoom` object with the specified dimensions and number of access points.
        """

        return OvalRoom(
            rows=rows, cols=cols, nr_access_points=nr_access_points, seed=self.room_seed
        )

    def create_donut_room(
        self, rows: int = None, cols: int = None, nr_access_points: int = 1
    ) -> "DonutRoom":
        """
        Creates a new DonutRoom object with the specified dimensions and access points.

        Args:
            rows (int): The number of rows in the room. Defaults to None.
            cols (int): The number of columns in the room. Defaults to None.
            nr_access_points (int): The number of access points in the room. Defaults to 1.

        Returns:
            DonutRoom: The newly created DonutRoom object.

        """

        min_ring_width = 1
        max_ring_width = min(rows, cols) // 1.75
        if max_ring_width <= min_ring_width:
            ring_width = min_ring_width
        else:
            ring_width = self.py_random.randint(min_ring_width, max_ring_width)

        ring_width = min_ring_width

        inner_shape = self.py_random.choice(["rectangle", "oval"])
        outer_shape = self.py_random.choice(["rectangle", "oval"])

        return DonutRoom(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            ring_width=ring_width,
            inner_shape=inner_shape,
            outer_shape=outer_shape,
            seed=self.room_seed,
        )

    def create_t_shape_room(
        self, rows: int = None, cols: int = None, nr_access_points: int = 1
    ) -> "TShapeRoom":
        """
        Creates a T-shaped room.

        Args:
            rows (int, optional): The number of rows in the room. Defaults to None.
            cols (int, optional): The number of columns in the room. Defaults to None.
            nr_access_points (int, optional): The number of access points in the room. Defaults to 1.

        Returns:
            TShapeRoom: The created T-shaped room instance.
        """

        horizontal_carve = 0.5 + self.py_random.random() * 0.5
        vertical_carve = 0.5 + self.py_random.random() * 0.5
        rotation = self.py_random.choice(["top", "right", "down", "left"])

        return TShapeRoom(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            horizontal_carve=horizontal_carve,
            vertical_carve=vertical_carve,
            rotation=rotation,
            seed=self.room_seed,
        )

    def create_l_shape_room(
        self, rows: int = None, cols: int = None, nr_access_points: int = 1
    ) -> "LShapeRoom":
        """
        Create an L-shape room.

        Args:
            rows (int, optional): The number of rows in the room. Defaults to None.
            cols (int, optional): The number of columns in the room. Defaults to None.
            nr_access_points (int, optional): The number of access points in the room. Defaults to 1.

        Returns:
            LShapeRoom: The created L-shape room.
        """

        horizontal_carve = 0.5 + self.py_random.random() * 0.5
        vertical_carve = 0.5 + self.py_random.random() * 0.5
        corner = self.py_random.choice(
            ["top_left", "top_right", "down_left", "down_right"]
        )
        return LShapeRoom(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            horizontal_carve=horizontal_carve,
            vertical_carve=vertical_carve,
            corner=corner,
            seed=self.room_seed,
        )

    def create_triangle_room(
        self, rows: int = None, cols: int = None, nr_access_points: int = 1
    ) -> "TriangleRoom":
        corner = self.py_random.choice(
            ["top_left", "top_right", "down_left", "down_right"]
        )
        return TriangleRoom(
            rows=rows,
            cols=cols,
            corner=corner,
            nr_access_points=nr_access_points,
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
        self.py_random = random.Random(self.seed)

        self.top_left_coord = top_left_coord
        self.down_right_coord = (self.rows, self.cols)
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

    def get_perimeter_cells(self, padding: int = 0) -> List[Tuple[int, int]]:
        """
        Generates a list of perimeter cells for the given grid, with an optional padding.
        Note: in the maze generation the mose interest is in the perimeter cells with padding 0, 1 and 2

        Args:
            padding (int, optional): The amount of padding to add around the grid. Defaults to 0.

        Returns:
            list: A list of perimeter cells as tuples (row, col).
        """
        room_mask = self.generate_inner_area_mask()

        # Always add a padding of 1 to ensure that rooms touching the edge are handled
        expanded_mask = np.pad(
            room_mask, padding + 1, mode="constant", constant_values=False
        )

        if padding == 0:
            perimeter_cells = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if expanded_mask[i + 1, j + 1] and self.is_perimeter(
                        expanded_mask, (i + 1, j + 1)
                    ):
                        perimeter_cells.append((i, j))
            return perimeter_cells

        struct = generate_binary_structure(2, 2)  # 2D square structuring element

        dilated_mask = binary_dilation(
            expanded_mask, structure=struct, iterations=padding
        )

        if padding == 1:
            perimeter_mask = dilated_mask & ~expanded_mask
        else:
            inner_dilated_mask = binary_dilation(
                expanded_mask, structure=struct, iterations=padding - 1
            )
            perimeter_mask = dilated_mask & ~inner_dilated_mask

        perimeter_cells = []
        for i in range(perimeter_mask.shape[0]):
            for j in range(perimeter_mask.shape[1]):
                if perimeter_mask[i, j]:
                    perimeter_cells.append((i - padding - 1, j - padding - 1))

        return perimeter_cells

    def is_perimeter(self, mask: np.ndarray, cell: Tuple[int, int]) -> bool:
        """
        Determines if a given cell is on the perimeter of a mask.

        Parameters:
            mask (np.ndarray): The mask representing the room.
            cell (Tuple[int, int]): The coordinates of the cell.

        Returns:
            bool: True if the cell is on the perimeter, False otherwise.
        """
        i, j = cell
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        if mask[i, j]:
            for ni, nj in neighbors:
                if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]:
                    if not mask[
                        ni, nj
                    ]:  # There's at least one neighboring cell not part of the room
                        return True
        return False

    def set_access_points(self) -> None:
        """
        Sets the access points for the object.

        This function selects a number of perimeter cells from the grid and assigns them as access points. The number of access points is determined by the `nr_access_points` attribute of the object.

        Parameters:
            None

        Returns:
            None
        """

        perimeter_cells = self.get_perimeter_cells()
        chosen_indices = self.np_random.choice(
            len(perimeter_cells), self.nr_access_points, replace=False
        )
        self.access_points = {perimeter_cells[i] for i in chosen_indices}

    def generate_inner_area_mask(self) -> np.ndarray:
        """
        Generate the inner area mask for the grid.
        The inner area represent all the cells that are withing the perimeter of the room
        (in case of a room with hollow areas this is not equal to the room mask!)

        Returns:
            np.ndarray: The inner area mask as a numpy array.
        """
        return (self.grid == PATH).astype(int)

    def get_non_perimeter_inner_cells(self) -> np.ndarray:
        """
        Returns a numpy array of non-perimeter inner cells.

        This function generates a binary mask of the inner area using the `generate_inner_area_mask` method. It then retrieves the list of perimeter cells using the `get_perimeter_cells` method with a padding of 0. The list of perimeter cells is then converted into a binary mask called `perimeter_mask` for easier subtraction. The `perimeter_mask` is subtracted from the `inner_mask` to obtain a mask where cells that are inside the inner area but not part of the perimeter are marked with 1s. The resulting mask is returned as a numpy array.

        Parameters:
            self (object): The instance of the class.

        Returns:
            np.ndarray: A numpy array of non-perimeter inner cells.
        """
        inner_mask = self.generate_inner_area_mask()

        # Get the list of perimeter cells.
        perimeter_cells = self.get_perimeter_cells(padding=0)

        # Convert the list of perimeter cells into a binary mask for easier subtraction.
        perimeter_mask = np.zeros_like(inner_mask)
        for cell in perimeter_cells:
            perimeter_mask[cell] = 1

        # Subtract the perimeter mask from the inner mask.
        # This will give a mask where cells that are inside the inner area but not part of the perimeter are marked with 1s.
        non_perimeter_mask = inner_mask - perimeter_mask

        return non_perimeter_mask


class RectangleRoom(Room):
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
        """Construct a RectangleRoom object representing a rectangle room.

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

    def generate_room_layout(self) -> np.ndarray:
        """
        Generate the room layout by setting all elements of the grid to PATH.

        Returns:
            List[List[str]]: The updated grid representing the room layout.
        """
        self.grid[:] = PATH
        return self.grid


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

    def generate_room_layout(self) -> np.ndarray:
        """
        Generates a room layout as a numpy array.

        Returns:
            np.ndarray: The generated room layout.
        """
        center_y, center_x = self.rows // 2, self.cols // 2
        a, b = center_x, center_y

        self.grid = generate_ellipse(
            self.rows,
            self.cols,
            a,
            b,
            center_x,
            center_y,
            value_true=PATH,
            value_false=WALL,
        )
        return self.grid


class DonutRoom(Room):
    """
    A donut room is a hollow area in the middle.
    The outer shape and the inner shape can be either rectangle or circular.
    Equal rows and columns are enforced, and we generally want a minimum ring width of 2.
    In the case with rectangle outer and inner shape, we can also get a ring
    width of 1, and we allow this.
    """

    def __init__(
        self,
        rows: int = 7,
        cols: int = 7,
        nr_access_points: int = 1,
        ring_width: int = 1,
        outer_shape: str = "oval",
        inner_shape: str = "oval",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Validation for shape
        valid_shapes = ["oval", "rectangle"]
        assert (
            outer_shape in valid_shapes
        ), f"Invalid outer_shape. Expected one of {valid_shapes}"
        assert (
            inner_shape in valid_shapes
        ), f"Invalid inner_shape. Expected one of {valid_shapes}"

        # Round down to odd number for better round shapes
        rows = rows - (rows % 2 == 0)
        cols = cols - (cols % 2 == 0)

        # Enforce same number of rows and cols
        rows = min(rows, cols)
        cols = min(rows, cols)

        super().__init__(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            min_rows=7,
            min_cols=7,
            seed=seed,
            **kwargs,
        )

        self.outer_shape = outer_shape
        self.inner_shape = inner_shape

        min_ring_width = 1
        if self.inner_shape == "oval":
            # special case when we need to enforce at least 2 width
            min_ring_width = 2

        max_ring_width = min(self.rows, self.cols) // 1.75
        self.ring_width = clamp(ring_width, min_ring_width, max_ring_width)

        self.generate_room_layout()
        self.set_access_points()

    def generate_room_layout(self)->np.ndarray:
        """
        Generates a room layout by creating a grid of cells based on the given parameters.
        
        Returns:
        - np.ndarray: The generated grid representing the room layout.
        """
        center_y, center_x = self.rows // 2, self.cols // 2
        a_outer, b_outer = center_x, center_y

        a_inner = a_outer - self.ring_width
        b_inner = b_outer - self.ring_width

        # Depending on the shape, generate the outer and inner masks
        self._outer_mask = self._generate_shape_mask(
            self.outer_shape, a_outer, b_outer, center_y, center_x, level="outer"
        )
        self._inner_mask = self._generate_shape_mask(
            self.inner_shape, a_inner, b_inner, center_y, center_x, level="inner"
        )

        self.grid = self._outer_mask - self._inner_mask
        return self.grid

    def _generate_shape_mask(self, shape: str, a: int, b: int, center_x: int, center_y: int, level: str="outer")->np.ndarray:
        """
        Generates a shape mask based on the given parameters.

        Args:
            shape (str): The shape of the mask to generate. Must be either "oval" or "rectangle".
            a (int): The length of the shape's horizontal axis.
            b (int): The length of the shape's vertical axis.
            center_x (int): The x-coordinate of the center of the shape.
            center_y (int): The y-coordinate of the center of the shape.
            level (str, optional): The level of the shape mask. Defaults to "outer".

        Returns:
            np.ndarray: The generated shape mask.
        """
        if a == 0 and b == 0:
            a, b = 1, 1

        if shape == "oval":
            return generate_ellipse(self.rows, self.cols, a, b, center_x, center_y)
        elif shape == "rectangle":
            return self.generate_rectangle_mask(a, b, center_x, center_y, level)

    def generate_rectangle_mask(self, a: int, b: int, center_x: int, center_y: int, level: str="outer")->np.ndarray:
        """
        Generate a rectangle mask with the specified dimensions and center position.

        Args:
            a (int): The width of the rectangle.
            b (int): The height of the rectangle.
            center_x (int): The x-coordinate of the center position.
            center_y (int): The y-coordinate of the center position.
            level (str, optional): The level of the mask. Defaults to "outer".

        Returns:
            np.ndarray: The generated rectangle mask.

        Note:
            - If the level is set to "inner", the width and height of the rectangle 
              are adjusted using a math solution.
            - If the outer shape is "oval" and the adjusted width and height are 
              greater than 1, the position of the top leftmost corner of the 
              rectangle is further adjusted.
        """
        if level == "inner":
            # math solution
            a, b = int(a / math.sqrt(2)), int(b / math.sqrt(2))

            # but need further adjust (at larger sizes the width of 2 is dangerous)
            if self.outer_shape == "oval":
                if a > 1 and b > 1:
                    # so find the position of the top leftmost corner of the outer mask
                    a -= 1
                    b -= 1

        mask = np.zeros((self.rows, self.cols), dtype=int)
        mask[center_y - b : center_y + b + 1, center_x - a : center_x + a + 1] = 1
        return mask

    def generate_inner_area_mask(self)->np.ndarray:
        """
        Generate the inner area mask based on the specified outer shape.

        Returns:
            np.ndarray: The inner area mask.
        """
        center_y, center_x = self.rows // 2, self.cols // 2
        a, b = center_x, center_y
        if self.outer_shape == "oval":
            return generate_ellipse(
                self.rows,
                self.cols,
                a,
                b,
                center_x,
                center_y,
                value_true=1,
                value_false=0,
            )

        elif self.outer_shape == "rectangle":
            mask = self.generate_rectangle_mask(a, b, center_x, center_y)
            return mask


class LShapeRoom(Room):
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        nr_access_points: int = 1,
        vertical_carve: float = 0.5,
        horizontal_carve: float = 0.5,
        corner: str = "top_left",
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            min_rows=4,
            min_cols=4,
            seed=seed,
            **kwargs,
        )
        assert 0 <= vertical_carve <= 1, "vertical_carve must be in the range [0, 1]"
        assert (
            0 <= horizontal_carve <= 1
        ), "horizontal_carve must be in the range [0, 1]"
        self.vertical_carve = vertical_carve
        self.horizontal_carve = horizontal_carve
        self.corner = corner
        self.generate_room_layout()
        self.set_access_points()

    def carve_amount(self, size: int, carve_ratio: float) -> int:
        """
        Carves out a certain amount from the given size based on the carve ratio.

        Parameters:
            size (int): The size of the input.
            carve_ratio (float): The ratio of the size to be carved out.

        Returns:
            int: The carved out amount from the size, with a minimum of 2.
        """
        return max(2, int(size * carve_ratio))

    def generate_room_layout(self)->np.ndarray:
        """
        Generates the layout of the room based on the given parameters.

        Parameters:
            self (object): The instance of the class.
        
        Returns:
            np.ndarray: The generated room layout as a NumPy array.
        """
        self.grid[:] = PATH  # Resetting the room to be full

        # Determine which carving ratio corresponds to vertical/horizontal carving
        vert_carve = self.carve_amount(self.rows - 2, self.vertical_carve)
        horz_carve = self.carve_amount(self.cols - 2, self.horizontal_carve)

        # Adjust carving based on desired corner
        if self.corner == "top_left":
            self.grid[:vert_carve, :horz_carve] = WALL
        elif self.corner == "top_right":
            self.grid[:vert_carve, -horz_carve:] = WALL
        elif self.corner == "down_left":
            self.grid[-vert_carve:, :horz_carve] = WALL
        elif self.corner == "down_right":
            self.grid[-vert_carve:, -horz_carve:] = WALL

        return self.grid


class TShapeRoom(Room):
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        nr_access_points: int = 1,
        vertical_carve: float = 0.5,
        horizontal_carve: float = 0.5,
        rotation: str = "top",
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            min_rows=4,
            min_cols=4,
            seed=seed,
            **kwargs,
        )
        assert 0 <= vertical_carve <= 1, "vertical_carve must be in the range [0, 1]"
        assert (
            0 <= horizontal_carve <= 1
        ), "horizontal_carve must be in the range [0, 1]"
        self.vertical_carve = vertical_carve
        self.horizontal_carve = horizontal_carve
        self.rotation = rotation
        super().__init__(
            rows=rows,
            cols=cols,
            nr_access_points=nr_access_points,
            min_rows=4,
            min_cols=4,
            seed=seed,
            **kwargs,
        )
        self.generate_room_layout()
        self.set_access_points()

    def carve_amount(self, size: int, carve_ratio: float) -> int:
        """
        Calculate the amount to carve based on the given size and carve ratio.

        Parameters:
            size (int): The size of the object to be carved.
            carve_ratio (float): The ratio of the object to be carved.

        Returns:
            int: The amount to carve, which is the maximum of 2 or the result of multiplying the size and carve ratio.
        """
        return max(2, int(size * carve_ratio))

    def generate_room_layout(self)->np.ndarray:
        """
        Generates the layout of the room based on the given rotation and carving ratios.

        Returns:
            np.ndarray: The generated room layout as a NumPy array.
        """
        self.grid[:] = PATH  # Resetting the room to be full

        # Based on the rotation, determine which carving ratio corresponds to vertical/horizontal carving
        if self.rotation in ["top", "down"]:
            vert_carve = self.carve_amount(self.rows - 2, self.vertical_carve)
            horz_carve = self.carve_amount(self.cols - 2, self.horizontal_carve)
        else:  # For left and right rotation, swap the carving ratios
            vert_carve = self.carve_amount(self.rows - 2, self.horizontal_carve)
            horz_carve = self.carve_amount(self.cols - 2, self.vertical_carve)

        # Adjust carving based on desired rotation
        if self.rotation == "top":
            carve_left = (self.cols - horz_carve) // 2
            carve_right = self.cols - carve_left
            self.grid[-vert_carve:, :carve_left] = WALL
            self.grid[-vert_carve:, carve_right:] = WALL
        elif self.rotation == "left":
            carve_top = (self.rows - vert_carve) // 2
            carve_down = self.rows - carve_top
            self.grid[:carve_top, -horz_carve:] = WALL
            self.grid[carve_down:, -horz_carve:] = WALL
        elif self.rotation == "right":
            carve_top = (self.rows - vert_carve) // 2
            carve_down = self.rows - carve_top
            self.grid[:carve_top, :horz_carve] = WALL
            self.grid[carve_down:, :horz_carve] = WALL
        elif self.rotation == "down":
            carve_left = (self.cols - horz_carve) // 2
            carve_right = self.cols - carve_left
            self.grid[:vert_carve, :carve_left] = WALL
            self.grid[:vert_carve, carve_right:] = WALL

        return self.grid


class TriangleRoom(Room):
    def __init__(self, rows: int = 4, cols: int = 4, corner="top_left", **kwargs):
        """
        Triangular Shaped Room class.

        Args:
            corner (str, optional): The corner of the rectangle to start carving from. Options are "top_left",
                                    "top_right", "down_left", "down_right". Defaults to "top_left".
        """
        super().__init__(rows=rows, cols=cols, min_rows=4, min_cols=4, **kwargs)
        self.corner = corner

        self.generate_room_layout()
        self.set_access_points()

    def generate_room_layout(self)->np.ndarray:
        """
        Generate the room layout based on the specified corner position.

        Returns:
            np.ndarray: The generated room layout as a 2D numpy array.
        """
        self.grid.fill(WALL)

        if self.corner == "top_right":
            self._carve_from_top_right()
        elif self.corner == "top_left":
            self._carve_from_top_left()
        elif self.corner == "down_right":
            self._carve_from_down_right()
        elif self.corner == "down_left":
            self._carve_from_down_left()

        return self.grid

    def _carve_from_top_right(self):
        col_ratio = self.cols / self.rows
        for i in range(self.rows):
            cols_to_carve = int((i + 1) * col_ratio)
            for j in range(cols_to_carve):
                self.grid[i][j] = PATH

    def _carve_from_top_left(self):
        col_ratio = self.cols / self.rows
        for i in range(self.rows):
            cols_to_carve = int((i + 1) * col_ratio)
            for j in range(self.cols - cols_to_carve, self.cols):
                self.grid[i][j] = PATH

    def _carve_from_down_right(self):
        col_ratio = self.cols / self.rows
        for i in range(self.rows):
            cols_to_carve = int((self.rows - i) * col_ratio)
            for j in range(cols_to_carve):
                self.grid[i][j] = PATH

    def _carve_from_down_left(self):
        col_ratio = self.cols / self.rows
        for i in range(self.rows):
            cols_to_carve = int((self.rows - i) * col_ratio)
            for j in range(self.cols - cols_to_carve, self.cols):
                self.grid[i][j] = PATH
