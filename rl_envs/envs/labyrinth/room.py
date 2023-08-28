from abc import ABC, abstractmethod
import math
import random
import numpy as np


class Room(ABC):
    def __init__(self, area=None, **kwargs):
        """The main way of building a shape will start from the desired area we want the room to have,
        but alternative constructors are possible."""
        self.area = area
        # If you have specific known attributes, consider unpacking them explicitly here.
        # For instance: self.width = kwargs.get("width", None)
        self.kwargs = kwargs

    @abstractmethod
    def get_shape(self):
        pass

    @abstractmethod
    def get_area(self):
        """Get the actual usable area of the room."""

        return self.area

    @abstractmethod
    def get_rectangular_footprint_area(self):
        """Get the area of the footprint of the room.

        The footprint is defined as the area of the rectangle
        with the same width and height as the room.
        """
        pass

    @abstractmethod
    def get_perimeter_cells(self):
        """Get a list of the coordinates of the perimeter cells."""
        pass

    @abstractmethod
    def get_room_mask(self):
        """Return a mask of the room.

        The mask should be a Boolean 2D array of the same shape as the room,
        with True showing where the room cells are, and False where the wall is.
        """
        pass

    @abstractmethod
    def set_access_points(self):
        """Define default access points for the room, based on its shape.
        
        Access points should be represented as a list of (x, y) tuples.
        """
        pass


class RectangularRoom(Room):
    def __init__(self, area=None, ratio=1, nr_access_points=1, **kwargs):
        # If area is specified, dynamically determine width and height
        if area:
            tentative_cols = int(math.sqrt(area / ratio))
            tentative_rows = int(ratio * tentative_cols)

            # If the result exceeds the area, adjust rows or cols.
            while tentative_rows * tentative_cols > area:
                if tentative_rows > tentative_cols:
                    tentative_rows -= 1
                else:
                    tentative_cols -= 1

            self.rows = tentative_rows
            self.cols = tentative_cols
        else: 
            self.rows = kwargs.get("rows", 10)  # Default values, can be changed
            self.cols = kwargs.get("cols", 10)

        super().__init__(area, **kwargs)

        self.nr_access_points = nr_access_points
        
        self.set_access_points()

    def get_shape(self):
        """Return the shape (rows, cols) of the room."""
        return self.rows, self.cols

    def get_area(self):
        """Get the actual usable area of the room."""
        return self.rows * self.cols

    def get_rectangular_footprint_area(self):
        """Get the area of the rectangular footprint of the room."""
        return self.get_area()

    def get_perimeter_cells(self):
        perimeter = []

        # Top and bottom walls
        for x in range(self.rows):
            perimeter.append((x, 0))
            perimeter.append((x, self.cols - 1))

        # Left and right walls, excluding corners
        for y in range(1, self.cols - 1):
            perimeter.append((0, y))
            perimeter.append((self.rows - 1, y))
            
        return perimeter

    def get_room_mask(self):
        """Return a mask of the room."""
        return [[True for _ in range(self.cols)] for _ in range(self.rows)]

    def set_access_points(self):
        """Define default access points for the room, based on its shape."""
        self.access_points = random.sample(self.get_perimeter_cells(), self.nr_access_points)

class CircularRoom(Room):
    # Implement similar methods for CircularRoom, approximating a circle shape on a grid.
    pass


class DonutRoom(Room):
    # Implement similar methods for DonutRoom, approximating a donut shape on a grid.
    pass


class LShapedRoom(Room):
    # Implement similar methods for LShapedRoom, approximating an L shape on a grid.
    pass
