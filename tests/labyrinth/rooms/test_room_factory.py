import pytest 
from rl_envs.envs.labyrinth.room import RoomFactory

class TestRoomFactory:
  
    def test_estimate_dimensions(self):
        factory = RoomFactory()
        rows, cols = factory._estimate_dimensions_from_area(100, ratio=1)
        assert rows * cols == 100

    def test_create_rectangular_room_from_dimensions(self):
        factory = RoomFactory()
        room = factory.create_rectangular_room(rows=5, cols=10)
        assert room.rows == 5
        assert room.cols == 10

    def test_create_rectangular_room_from_area(self):
        factory = RoomFactory()
        rows, cols = factory._estimate_dimensions_from_area(100, ratio=1)
        room = factory.create_rectangular_room(rows, cols)
        assert room.rows * room.cols == 100

    def test_create_rectangular_room_from_area_randomized_ratios(self):
        factory = RoomFactory(ratio_range=(0.3, 3))

        for _ in range(10):
            room = factory.create_room(desired_area=100)
            assert room.rows * room.cols <= 100
    
    def test_room_area_too_small(self):
        factory = RoomFactory(ratio_range=(0.3, 3))
        with pytest.raises(ValueError):
            room = factory.create_room(desired_area=1)
