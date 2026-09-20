import numpy as np

from precompute_potential_maps import _tile_easy_geometry_xy


def test_tile_easy_geometry_xy_creates_eight_surrounding_copies():
    geom = {
        "map_x_max": 10.0,
        "map_y_min": -8.0,
        "map_y_max": 8.0,
        "map_length": 16.0,
        "map_meta": {"map_type": "easy"},
        "spawn_start": (5.0, -7.45, 2.5),
        "spawn_goal": (5.0, 7.45, 2.5),
        "balls": np.asarray([[2.0, 3.0, 2.5, 0.4]], dtype=np.float32),
        "cyl": np.asarray([[4.0, -1.0, 0.2]], dtype=np.float32),
        "voxels": np.asarray([[6.0, 2.0, 1.0, 0.3, 0.4, 0.5]], dtype=np.float32),
        "cyl_h": np.empty((0, 3), dtype=np.float32),
    }

    tiled = _tile_easy_geometry_xy(geom, tile_radius=1)

    assert tiled["balls"].shape == (9, 4)
    assert tiled["cyl"].shape == (9, 3)
    assert tiled["voxels"].shape == (9, 6)
    assert tiled["map_x_min"] == -10.0
    assert tiled["map_x_max"] == 20.0
    assert tiled["map_y_min"] == -24.0
    assert tiled["map_y_max"] == 24.0
    assert tiled["map_length"] == 48.0
    assert tiled["spawn_start"] == geom["spawn_start"]
    assert tiled["spawn_goal"] == geom["spawn_goal"]
    assert tiled["map_meta"]["xy_tiling"] == "3x3"

    ball_xy = {tuple(row[:2]) for row in tiled["balls"]}
    assert (2.0, 3.0) in ball_xy
    assert (-8.0, -13.0) in ball_xy
    assert (12.0, 19.0) in ball_xy


def test_zero_tile_radius_keeps_original_geometry():
    geom = {"sentinel": object()}
    assert _tile_easy_geometry_xy(geom, tile_radius=0) is geom
