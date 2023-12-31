import pytest
from rl_envs_forge.envs.labyrinth.maze.room import RoomFactory


class TestRoomFactory:
    def test_estimate_dimensions(self):
        factory = RoomFactory()
        rows, cols = factory._estimate_dimensions_from_area(100, ratio=1)
        assert rows * cols == 100

    def test_create_rectangle_room_from_dimensions(self):
        factory = RoomFactory()
        room = factory.create_rectangle_room(rows=5, cols=10)
        assert room.rows == 5
        assert room.cols == 10

    def test_create_rectangle_room_from_area(self):
        factory = RoomFactory()
        rows, cols = factory._estimate_dimensions_from_area(100, ratio=1)
        room = factory.create_rectangle_room(rows, cols)
        assert room.rows * room.cols == 100

    def test_create_rectangle_room_from_area_randomized_ratios(self):
        factory = RoomFactory(ratio_range=(0.3, 3))

        for _ in range(10):
            room = factory.create_room(desired_area=100)
            assert room.rows * room.cols <= 100

    def test_room_area_too_small(self):
        factory = RoomFactory(ratio_range=(0.3, 3))
        with pytest.raises(ValueError):
            room = factory.create_room(desired_area=1)

    def test_fixed_access_points_and_ratio(self):
        factory = RoomFactory(access_points_nr=3, ratio=0.5, room_types=["rectangle"])
        room = factory.create_room(desired_area=100)
        assert len(room.access_points) == 3
        assert room.rows == room.cols / 2

    def test_unknown_room_type(self):
        factory = RoomFactory(room_types=["unknown"])
        with pytest.raises(ValueError):
            room = factory.create_room(desired_area=100)
