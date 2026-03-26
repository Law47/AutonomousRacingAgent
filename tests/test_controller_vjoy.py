import numpy as np

from Common.controller_vjoy import clamp_controls, convert_to_vjoy_axes


def test_clamp_controls_uses_3d_action_ranges():
    steer, accel, brake = clamp_controls(2.5, -1.0, 4.0)
    assert steer == 1.0
    assert accel == 0.0
    assert brake == 1.0


def test_convert_to_vjoy_axes_matches_expected_scale():
    axis_x, axis_y, axis_z = convert_to_vjoy_axes(0.0, 0.5, 1.0)
    assert axis_x == 16384
    assert axis_y == 16384
    assert axis_z == 32768
