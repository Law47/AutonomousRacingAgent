import sys
import types
import pickle
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ASSETTO_GYM = ROOT / "assetto_corsa_gym"
if str(ASSETTO_GYM) not in sys.path:
    sys.path.insert(0, str(ASSETTO_GYM))
DISCOR = ROOT / "algorithm" / "discor"
if str(DISCOR) not in sys.path:
    sys.path.insert(0, str(DISCOR))

try:
    from gym.spaces import Box
except ModuleNotFoundError:
    class Box:
        def __init__(self, low, high):
            self.low = low
            self.high = high
            self.shape = low.shape

    class Env:
        pass

    class EzPickle:
        def __init__(self, *args, **kwargs):
            pass

    gym_module = types.ModuleType("gym")
    gym_spaces_module = types.ModuleType("gym.spaces")
    gym_utils_module = types.ModuleType("gym.utils")
    gym_spaces_module.Box = Box
    gym_utils_module.EzPickle = EzPickle
    gym_module.Env = Env
    gym_module.spaces = gym_spaces_module
    gym_module.utils = gym_utils_module
    sys.modules["gym"] = gym_module
    sys.modules["gym.spaces"] = gym_spaces_module
    sys.modules["gym.utils"] = gym_utils_module

ac_client_module = types.ModuleType("AssettoCorsaEnv.ac_client")
track_module = types.ModuleType("AssettoCorsaEnv.track")
reference_lap_module = types.ModuleType("AssettoCorsaEnv.reference_lap")
sensors_module = types.ModuleType("AssettoCorsaEnv.sensors_ray_casting")
gap_module = types.ModuleType("AssettoCorsaEnv.gap")
brake_map_module = types.ModuleType("AssettoCorsaEnv.brake_map")


class Client:
    pass


class Track:
    pass


class ReferenceLap:
    pass


class SensorsRayCasting:
    pass


class BrakeMap:
    pass


def get_gap(*args, **kwargs):
    return None


ac_client_module.Client = Client
track_module.Track = Track
reference_lap_module.ReferenceLap = ReferenceLap
sensors_module.SensorsRayCasting = SensorsRayCasting
sensors_module.MAX_RAY_LEN = 1.0
gap_module.get_gap = get_gap
brake_map_module.BrakeMap = BrakeMap
sys.modules["AssettoCorsaEnv.ac_client"] = ac_client_module
sys.modules["AssettoCorsaEnv.track"] = track_module
sys.modules["AssettoCorsaEnv.reference_lap"] = reference_lap_module
sys.modules["AssettoCorsaEnv.sensors_ray_casting"] = sensors_module
sys.modules["AssettoCorsaEnv.gap"] = gap_module
sys.modules["AssettoCorsaEnv.brake_map"] = brake_map_module

from AssettoCorsaEnv.ac_env import AssettoCorsaEnv, GearShiftGate, STREAMING_DEMO_FORMAT
from AssettoCorsaEnv.data_loader import DataLoader
from AssettoCorsaEnv.gear_shift_labels import infer_shift_from_state
from AssettoCorsaEnv import vjoy as vjoy_module
from AssettoCorsaPlugin.plugins.sensors_par import car_control
from discor.replay_buffer import ReplayBuffer
from discor.algorithm.sac import SAC


class FakeControls(dict):
    def set_controls(self, **kwargs):
        self.update(kwargs)


class FakeClient:
    def __init__(self):
        self.controls = FakeControls()
        self.responded = False

    def respond_to_server(self):
        self.responded = True


def make_env_stub(cooldown_steps=0):
    env = AssettoCorsaEnv.__new__(AssettoCorsaEnv)
    env.control_action_dim = 3
    env.action_dim = 5
    env.action_space = Box(
        low=np.full((env.action_dim,), -1.0, dtype=np.float32),
        high=np.full((env.action_dim,), 1.0, dtype=np.float32),
    )
    env.use_relative_actions = True
    env.adjusted_controls_rate_limit = np.array(
        [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float32,
    )
    env.controls_min_values = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    env.controls_max_values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    env.current_actions = np.array([0.0, -1.0, -1.0], dtype=np.float32)
    env.shift_gate = GearShiftGate(cooldown_steps=cooldown_steps)
    env.shift_up_count = 0
    env.shift_down_count = 0
    env.use_reference_line_in_reward = False
    env.penalize_actions_diff = False
    env.reverse_gear_step_penalty = 0.001
    env.enable_out_of_track_penalty = False
    env.out_of_track_penalty_start = 0.1
    env.out_of_track_penalty_end = 0.1
    env.out_of_track_penalty_ramp_steps = 0
    env.total_steps = 0
    env.client = FakeClient()
    return env


def test_shift_gate_below_threshold_emits_no_pulse():
    gate = GearShiftGate(cooldown_steps=0)

    assert gate.update(0.39, 0.0) == (False, False)


def test_shift_gate_crossing_emits_one_pulse_until_rearmed():
    gate = GearShiftGate(cooldown_steps=0)

    assert gate.update(0.41, 0.0) == (True, False)
    assert gate.update(1.0, 0.0) == (False, False)
    assert gate.update(0.04, 0.0) == (False, False)
    assert gate.update(0.9, 0.0) == (True, False)


def test_shift_gate_cooldown_blocks_repeated_shift():
    gate = GearShiftGate(cooldown_steps=2)

    assert gate.update(0.9, 0.0) == (True, False)
    assert gate.update(0.0, 0.0) == (False, False)
    assert gate.update(0.9, 0.0) == (False, False)
    assert gate.update(0.0, 0.0) == (False, False)
    assert gate.update(0.9, 0.0) == (True, False)


def test_shift_gate_simultaneous_crossing_suppresses_both():
    gate = GearShiftGate(cooldown_steps=0)

    assert gate.update(0.9, 0.9) == (False, False)


def test_env_action_space_and_preprocess_are_split():
    env = make_env_stub()
    action = np.array([0.5, 0.25, 0.75, 0.9, -0.9], dtype=np.float32)

    controls = env.preprocess_actions(action, env.current_actions)

    assert env.action_space.shape == (5,)
    assert controls.shape == (3,)
    np.testing.assert_allclose(controls, [0.5, -0.75, -0.25])


def test_env_set_actions_forwards_decoded_shift_pulse():
    env = make_env_stub()

    env.set_actions(np.array([0.0, 0.0, 0.0, 0.9, 0.0], dtype=np.float32))

    assert env.client.responded is True
    assert env.client.controls["enable_gear_shift"] is True
    assert env.client.controls["shift_up"] is True
    assert env.client.controls["shift_down"] is False
    np.testing.assert_allclose(env.raw_actions, [0.0, 0.0, 0.0, 0.9, 0.0])


def test_offline_loader_pads_old_three_control_actions_to_model_shape():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 5})()

    padded = loader.pad_model_actions(np.array([0.1, -0.2, 0.3], dtype=np.float32))

    np.testing.assert_allclose(padded, [0.1, -0.2, 0.3, 0.0, 0.0])


def test_offline_loader_prefers_recorded_five_action_demo_tensor():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 5})()

    recorded = loader.get_recorded_model_actions(
        {"actions_0": 0.1, "actions_1": -0.2, "actions_2": 0.3, "actions_3": 1.0, "actions_4": 0.0}
    )

    np.testing.assert_allclose(recorded, [0.1, -0.2, 0.3, 1.0, 0.0])


def test_offline_loader_infers_shift_up_from_gear_delta():
    loader = DataLoader.__new__(DataLoader)
    loader.shift_label_min_drive_gear = 2

    inferred = loader.infer_shift_actions({"actualGear": 4}, {"actualGear": 3})

    np.testing.assert_allclose(inferred, [1.0, 0.0])


def test_stable_shift_labels_ignore_transient_neutral_gear():
    previous_stable_gear = None
    sequence = [6, 1, 7]
    labels = []

    for gear in sequence:
        shift_actions, previous_stable_gear, _ = infer_shift_from_state(
            {"actualGear": gear},
            previous_stable_gear,
            min_drive_gear=2,
        )
        labels.append(shift_actions)

    np.testing.assert_allclose(labels[0], [0.0, 0.0])
    np.testing.assert_allclose(labels[1], [0.0, 0.0])
    np.testing.assert_allclose(labels[2], [1.0, 0.0])


def test_offline_loader_replaces_bad_recorded_shift_labels():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 5, "gear_shift_threshold": 0.5})()

    actions, changed = loader.replace_shift_labels(
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )

    assert changed is True
    np.testing.assert_allclose(actions, [0.0, 0.0, 0.0, 0.0, 0.0])


def test_demo_shift_alignment_matches_model_shift_gate_indices():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 5, "gear_shift_threshold": 0.5})()
    loader.shift_label_min_drive_gear = 2
    trajectory = [
        {"actualGear": 2, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 0.0, "actions_4": 0.0},
        {"actualGear": 3, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 1.0, "actions_4": 0.0},
        {"actualGear": 2, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 0.0, "actions_4": 1.0},
    ]

    stats = loader.validate_shift_action_alignment(trajectory)

    assert stats["gear_up_events"] == 1
    assert stats["gear_down_events"] == 1
    assert stats["shift_up_signals"] == 1
    assert stats["shift_down_signals"] == 1
    assert stats["mismatches"] == 0


def test_demo_shift_alignment_detects_missing_shift_signal():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 5, "gear_shift_threshold": 0.5})()
    loader.shift_label_min_drive_gear = 2
    trajectory = [
        {"actualGear": 2, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 0.0, "actions_4": 0.0},
        {"actualGear": 3, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 0.0, "actions_4": 0.0},
    ]

    stats = loader.validate_shift_action_alignment(trajectory)

    assert stats["gear_up_events"] == 1
    assert stats["shift_up_signals"] == 0
    assert stats["mismatches"] == 1


def test_demo_shift_alignment_flags_neutral_transition_shift_noise():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 5, "gear_shift_threshold": 0.5})()
    loader.shift_label_min_drive_gear = 2
    trajectory = [
        {"actualGear": 6, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 0.0, "actions_4": 0.0},
        {"actualGear": 1, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 0.0, "actions_4": 1.0},
        {"actualGear": 7, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 1.0, "actions_4": 0.0},
    ]

    stats = loader.validate_shift_action_alignment(trajectory)

    assert stats["gear_up_events"] == 1
    assert stats["gear_down_events"] == 0
    assert stats["ignored_unstable_gear_frames"] == 1
    assert stats["mismatches"] == 1


def test_replay_buffer_iter_batches_visits_demo_dataset_once():
    buffer = ReplayBuffer(memory_size=5, state_shape=(1,), action_shape=(5,), gamma=0.99, nstep=1)
    for index in range(5):
        state = np.array([index], dtype=np.float32)
        action = np.full((5,), index, dtype=np.float32)
        buffer.append(state, action, float(index), state + 1.0, False)

    batches = list(buffer.iter_batches(batch_size=2, num_samples=5, shuffle=False))

    batch_sizes = [batch[0].shape[0] for batch in batches]
    visited_states = np.concatenate([batch[0].cpu().numpy().reshape(-1) for batch in batches])
    assert batch_sizes == [2, 2, 1]
    np.testing.assert_allclose(visited_states, [0, 1, 2, 3, 4])


def test_sac_behavior_cloning_trains_against_demo_actions():
    class FakeWriter:
        def add_scalar(self, *args, **kwargs):
            pass

    algo = SAC(
        state_dim=4,
        action_dim=5,
        device=torch.device("cpu"),
        policy_hidden_units=[8],
        q_hidden_units=[8],
        log_interval=1,
        seed=0,
    )
    states = torch.zeros((4, 4), dtype=torch.float32)
    demo_actions = torch.zeros((4, 5), dtype=torch.float32)
    demo_actions[:, 3] = 1.0
    rewards = torch.zeros((4, 1), dtype=torch.float32)
    dones = torch.zeros((4, 1), dtype=torch.float32)
    batch = (states, demo_actions, rewards, states, dones)

    stats = algo.update_behavior_cloning(
        batch,
        FakeWriter(),
        action_weights=np.array([1.0, 1.0, 1.0, 25.0, 25.0], dtype=np.float32),
        loss_coef=1.0,
    )

    assert stats["behavior_clone_loss"] > 0.0


def test_load_history_reads_streaming_demo_file_and_ignores_partial_tail(tmp_path):
    env = AssettoCorsaEnv.__new__(AssettoCorsaEnv)
    save_path = tmp_path / "demo.pkl"
    static_info = {"TrackFullName": "monza", "CarName": "bmw_z4_gt3"}
    states = [{"step": 1}, {"step": 2}, {"step": 3}]

    with open(save_path, "wb") as file_handle:
        pickle.dump({"format": STREAMING_DEMO_FORMAT, "static_info": static_info}, file_handle)
        pickle.dump({"states": states[:2]}, file_handle)
        pickle.dump({"states": states[2:]}, file_handle)
        truncated_chunk = pickle.dumps({"states": [{"step": 999}]}, protocol=pickle.HIGHEST_PROTOCOL)[:-7]
        file_handle.write(truncated_chunk)

    trajectory, loaded_static_info = env.load_history(save_path)

    assert loaded_static_info == static_info
    assert trajectory == states


def test_gear_shift_reward_does_not_reward_upshift_at_high_rpm():
    env = make_env_stub()
    state = {"RPM": 8000.0, "actualGear": 3, "shift_up": True, "shift_down": False}

    assert env.get_gear_shift_reward(state) == 0.0
    assert state["gear_shift_reward"] == 0.0


def test_gear_shift_reward_does_not_reward_downshift_at_low_rpm():
    env = make_env_stub()
    state = {"RPM": 2500.0, "actualGear": 3, "shift_up": False, "shift_down": True}

    assert env.get_gear_shift_reward(state) == 0.0


def test_gear_shift_reward_does_not_penalize_wrong_rpm_shift():
    env = make_env_stub()
    state = {"RPM": 3500.0, "actualGear": 3, "shift_up": True, "shift_down": False}

    assert env.get_gear_shift_reward(state) == 0.0


def test_gear_shift_reward_does_not_penalize_neutral():
    env = make_env_stub()
    state = {"RPM": 4000.0, "actualGear": 0, "shift_up": False, "shift_down": False}

    assert env.get_gear_shift_reward(state) == 0.0
    assert state["reverse_gear_penalty"] == 0.0


def test_gear_shift_reward_penalizes_reverse():
    env = make_env_stub()
    state = {"RPM": 4000.0, "actualGear": -1, "shift_up": False, "shift_down": False}

    assert env.get_gear_shift_reward(state) == -0.001
    assert state["reverse_gear_penalty"] == -0.001


def test_gear_shift_reward_does_not_reward_shift_signal_before_threshold():
    env = make_env_stub()
    state = {
        "RPM": 8000.0,
        "actualGear": 3,
        "shift_up": False,
        "shift_down": False,
        "actions_3": 0.5,
        "actions_4": 0.0,
    }

    assert env.get_gear_shift_reward(state) == 0.0


def test_out_of_track_penalty_applies_only_when_enabled():
    env = make_env_stub()
    state = {"out_of_track": 1.0}

    assert env.get_out_of_track_penalty(state) == 0.0
    assert state["out_of_track_penalty"] == 0.0

    env.enable_out_of_track_penalty = True

    assert env.get_out_of_track_penalty(state) == -0.1
    assert state["out_of_track_penalty"] == -0.1


def test_out_of_track_penalty_ramps_with_training_steps():
    env = make_env_stub()
    env.enable_out_of_track_penalty = True
    env.out_of_track_penalty_start = 0.075
    env.out_of_track_penalty_end = 0.5
    env.out_of_track_penalty_ramp_steps = 3_000_000

    env.total_steps = 0
    assert np.isclose(env.get_current_out_of_track_penalty(), 0.075)

    env.total_steps = 1_500_000
    assert np.isclose(env.get_current_out_of_track_penalty(), 0.2875)

    env.total_steps = 3_000_000
    assert np.isclose(env.get_current_out_of_track_penalty(), 0.5)


def test_vigem_backend_maps_shift_up_to_a_and_shift_down_to_x(monkeypatch):
    class FakeGamepad:
        def __init__(self):
            self.pressed = []
            self.released = []

        def left_joystick_float(self, **kwargs):
            pass

        def right_trigger_float(self, **kwargs):
            pass

        def left_trigger_float(self, **kwargs):
            pass

        def press_button(self, button):
            self.pressed.append(button)

        def release_button(self, button):
            self.released.append(button)

        def update(self):
            pass

    class FakeVG:
        XUSB_BUTTON = type(
            "XUSB_BUTTON",
            (),
            {"XUSB_GAMEPAD_A": "A", "XUSB_GAMEPAD_X": "X"},
        )
        gamepad = None

        @staticmethod
        def VX360Gamepad():
            FakeVG.gamepad = FakeGamepad()
            return FakeVG.gamepad

    monkeypatch.setattr(car_control, "vg", FakeVG)
    controls = car_control.Controls(backend="vigem")

    controls.set_controls(0.0, -1.0, -1.0, enable_gear_shift=True, shift_up=True)
    assert "A" in FakeVG.gamepad.pressed
    controls.set_controls(0.0, -1.0, -1.0, enable_gear_shift=True)
    controls.set_controls(0.0, -1.0, -1.0, enable_gear_shift=True)
    assert FakeVG.gamepad.pressed.count("A") == car_control.SHIFT_BUTTON_HOLD_UPDATES
    controls.set_controls(0.0, -1.0, -1.0, enable_gear_shift=True)
    assert "A" in FakeVG.gamepad.released

    controls.set_controls(0.0, -1.0, -1.0, enable_gear_shift=True, shift_down=True)
    assert "X" in FakeVG.gamepad.pressed
    assert "A" in FakeVG.gamepad.released


def test_vjoy_backend_maps_shift_down_to_button_three(monkeypatch):
    class FakeVJoy:
        instance = None

        def __init__(self):
            FakeVJoy.instance = self
            self.positions = []

        def open(self):
            return True

        def close(self):
            return True

        def generateJoystickPosition(self, **kwargs):
            return kwargs

        def update(self, joystickPosition):
            self.positions.append(joystickPosition)
            return True

    monkeypatch.setattr(car_control, "vJoy", FakeVJoy)
    controls = car_control.Controls(backend="vjoy")

    controls.set_controls(0.0, -1.0, -1.0, enable_gear_shift=True, shift_up=True)
    assert FakeVJoy.instance.positions[-1]["lButtons"] == car_control.VJOY_SHIFT_UP_BUTTON
    controls.set_controls(0.0, -1.0, -1.0, enable_gear_shift=True)
    assert FakeVJoy.instance.positions[-1]["lButtons"] == car_control.VJOY_SHIFT_UP_BUTTON

    controls.set_controls(0.0, -1.0, -1.0, enable_gear_shift=True, shift_down=True)
    assert FakeVJoy.instance.positions[-1]["lButtons"] == car_control.VJOY_SHIFT_DOWN_BUTTON


def test_vjoy_backend_accepts_configured_dll_path(monkeypatch, tmp_path):
    class FakeVJoy:
        def open(self):
            return True

        def close(self):
            return True

        def generateJoystickPosition(self, **kwargs):
            return kwargs

        def update(self, joystickPosition):
            return True

    dll_path = tmp_path / "vJoyInterface.dll"
    dll_path.write_bytes(b"fake")
    monkeypatch.delenv("VJOY_DLL_PATH", raising=False)
    monkeypatch.setattr(car_control, "vJoy", FakeVJoy)

    car_control.Controls(backend="vjoy", vjoy_dll_path=str(dll_path))

    assert os.environ["VJOY_DLL_PATH"] == str(dll_path)


def test_vjoy_dll_resolver_uses_env_var_path(tmp_path, monkeypatch):
    dll_path = tmp_path / "vJoyInterface.dll"
    dll_path.write_bytes(b"fake")
    monkeypatch.setenv("VJOY_DLL_PATH", str(dll_path))

    assert vjoy_module.resolve_vjoy_dll_path() == str(dll_path)
