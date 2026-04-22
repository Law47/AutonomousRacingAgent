import sys
import types
import pickle
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

from AssettoCorsaEnv.ac_env import AssettoCorsaEnv, ShiftExecutionGate, STREAMING_DEMO_FORMAT
from AssettoCorsaEnv.autoshift import AutoShifter
from AssettoCorsaEnv.data_loader import DataLoader
from AssettoCorsaPlugin.plugins.sensors_par import car_control
from discor.agent import Agent, TRAINING_STATE_FILENAME
from discor.replay_buffer import ReplayBuffer


class FakeControls(dict):
    def set_controls(self, **kwargs):
        self.update(kwargs)


class FakeClient:
    def __init__(self):
        self.controls = FakeControls()
        self.responded = False
        self.state = {}

    def respond_to_server(self):
        self.responded = True


class DummySpace:
    def __init__(self, shape):
        self.shape = shape

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


class DummyAgentEnv:
    observation_space = DummySpace((2,))
    action_space = DummySpace((3,))

    def seed(self, seed):
        self.seed_value = seed


class DummyAlgo:
    gamma = 0.99
    nstep = 1

    def save_models(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.marker").write_text("ok", encoding="utf-8")

    def load_models(self, path):
        assert (Path(path) / "model.marker").exists()


def make_env_stub(cooldown_steps=0):
    env = AssettoCorsaEnv.__new__(AssettoCorsaEnv)
    env.control_action_dim = 3
    env.shift_action_dim = 2
    env.action_dim = 3
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
    env.shift_gate = ShiftExecutionGate(cooldown_steps=cooldown_steps)
    env.shift_up_count = 0
    env.shift_down_count = 0
    env.enforce_mutually_exclusive_pedals = True
    env.prevent_reverse_downshift = True
    env.auto_shifter = type("AutoShiftStub", (), {"gear_index_offset": 0})()
    env.use_reference_line_in_reward = False
    env.penalize_actions_diff = False
    env.total_steps = 0
    env.client = FakeClient()
    env.raw_actions = np.zeros(env.control_action_dim, dtype=np.float32)
    env.raw_shift_actions = np.zeros(env.shift_action_dim, dtype=np.float32)
    env.shift_teacher_actions = np.zeros(env.shift_action_dim, dtype=np.float32)
    env.shift_source = "manual"
    return env


def append_agent_transition(buffer, index):
    state = np.array([index, index + 0.1], dtype=np.float32)
    action = np.array([index, -index, index + 0.5], dtype=np.float32)
    shift_label = np.array([index % 2, (index + 1) % 2], dtype=np.float32)
    next_state = state + 10.0
    buffer.append(
        state,
        action,
        float(index),
        next_state,
        bool(index % 2),
        shift_label=shift_label,
    )


def test_shift_execution_below_threshold_emits_no_pulse():
    gate = ShiftExecutionGate(cooldown_steps=0)

    assert gate.update(0.39, 0.0) == (False, False)


def test_shift_execution_repeats_bernoulli_events_without_release_gate():
    gate = ShiftExecutionGate(cooldown_steps=0)

    assert gate.update(0.51, 0.0) == (True, False)
    assert gate.update(1.0, 0.0) == (True, False)


def test_shift_execution_cooldown_blocks_repeated_shift():
    gate = ShiftExecutionGate(cooldown_steps=2)

    assert gate.update(0.9, 0.0) == (True, False)
    assert gate.update(0.9, 0.0) == (False, False)
    assert gate.update(0.9, 0.0) == (False, False)
    assert gate.update(0.9, 0.0) == (True, False)


def test_shift_execution_simultaneous_crossing_suppresses_both():
    gate = ShiftExecutionGate(cooldown_steps=0)

    assert gate.update(0.9, 0.9) == (False, False)


def test_env_action_space_and_preprocess_are_split():
    env = make_env_stub()
    action = np.array([0.5, 0.25, 0.75], dtype=np.float32)

    controls = env.preprocess_actions(action, env.current_actions)

    assert env.action_space.shape == (3,)
    assert controls.shape == (3,)
    np.testing.assert_allclose(controls, [0.5, -1.0, -0.25])


def test_env_preprocess_keeps_only_more_extreme_pedal():
    env = make_env_stub()
    env.current_actions = np.array([0.0, -1.0, -1.0], dtype=np.float32)

    controls = env.preprocess_actions(np.array([0.0, 1.0, 0.2], dtype=np.float32), env.current_actions)

    np.testing.assert_allclose(controls, [0.0, 0.0, -1.0])


def test_env_set_actions_forwards_decoded_shift_pulse():
    env = make_env_stub()
    env.client.state["actualGear"] = 3

    env.set_actions(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        shift_action=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert env.client.responded is True
    assert env.client.controls["enable_gear_shift"] is True
    assert env.client.controls["shift_up"] is True
    assert env.client.controls["shift_down"] is False
    np.testing.assert_allclose(env.raw_actions, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(env.raw_shift_actions, [1.0, 0.0])


def test_env_set_actions_blocks_downshift_into_reverse():
    env = make_env_stub()
    env.client.state["actualGear"] = 1

    env.set_actions(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        shift_action=np.array([0.0, 1.0], dtype=np.float32),
    )

    assert env.client.controls["shift_up"] is False
    assert env.client.controls["shift_down"] is False


def test_env_set_actions_blocks_downshift_into_neutral_with_gear_offset():
    env = make_env_stub()
    env.auto_shifter.gear_index_offset = 1
    env.client.state["actualGear"] = 2

    env.set_actions(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        shift_action=np.array([0.0, 1.0], dtype=np.float32),
    )

    assert env.client.controls["shift_up"] is False
    assert env.client.controls["shift_down"] is False


def test_offline_loader_keeps_three_control_actions():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 3})()

    padded = loader.pad_model_actions(np.array([0.1, -0.2, 0.3], dtype=np.float32))

    np.testing.assert_allclose(padded, [0.1, -0.2, 0.3])


def test_offline_loader_prefers_recorded_five_action_demo_tensor():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 3, "control_action_dim": 3, "shift_action_dim": 2})()

    recorded = loader.get_recorded_model_actions(
        {"actions_0": 0.1, "actions_1": -0.2, "actions_2": 0.3, "actions_3": 1.0, "actions_4": 0.0}
    )

    np.testing.assert_allclose(recorded, [0.1, -0.2, 0.3, 1.0, 0.0])


def test_offline_loader_rebuilds_controls_but_preserves_recorded_shift_signals():
    loader = DataLoader.__new__(DataLoader)
    loader.prev_abs_actions = np.array([0.0, -1.0, -1.0], dtype=np.float32)
    loader.env = type(
        "EnvStub",
        (),
        {
            "action_dim": 3,
            "control_action_dim": 3,
            "shift_action_dim": 2,
            "inverse_preprocess_actions": lambda self, prev_abs, current_abs: np.array([0.2, -1.0, 0.4], dtype=np.float32),
        },
    )()

    actions, shift_label = loader.compose_transition_labels(
        {"actions_0": 0.9, "actions_1": 0.9, "actions_2": 0.9, "actions_3": 1.0, "actions_4": 0.0},
        {"actualGear": 2},
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(actions, [0.2, -1.0, 0.4])
    np.testing.assert_allclose(shift_label, [1.0, 0.0])


def test_offline_loader_infers_shift_up_from_gear_delta():
    loader = DataLoader.__new__(DataLoader)

    inferred = loader.infer_shift_actions({"actualGear": 4}, {"actualGear": 3})

    np.testing.assert_allclose(inferred, [1.0, 0.0])


def test_demo_shift_alignment_matches_shift_execution_indices():
    loader = DataLoader.__new__(DataLoader)
    loader.env = type("EnvStub", (), {"action_dim": 3, "control_action_dim": 3, "shift_action_dim": 2, "shift_execution_threshold": 0.5})()
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
    loader.env = type("EnvStub", (), {"action_dim": 3, "control_action_dim": 3, "shift_action_dim": 2, "shift_execution_threshold": 0.5})()
    trajectory = [
        {"actualGear": 2, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 0.0, "actions_4": 0.0},
        {"actualGear": 3, "actions_0": 0.0, "actions_1": 0.0, "actions_2": 0.0, "actions_3": 0.0, "actions_4": 0.0},
    ]

    stats = loader.validate_shift_action_alignment(trajectory)

    assert stats["gear_up_events"] == 1
    assert stats["shift_up_signals"] == 0
    assert stats["mismatches"] == 1


def test_replay_buffer_iter_batches_visits_demo_dataset_once():
    buffer = ReplayBuffer(memory_size=5, state_shape=(1,), action_shape=(3,), gamma=0.99, nstep=1)
    for index in range(5):
        state = np.array([index], dtype=np.float32)
        action = np.full((3,), index, dtype=np.float32)
        shift_label = np.array([index % 2, 0.0], dtype=np.float32)
        buffer.append(state, action, float(index), state + 1.0, False, shift_label=shift_label)

    batches = list(buffer.iter_batches(batch_size=2, num_samples=5, shuffle=False))

    batch_sizes = [batch[0].shape[0] for batch in batches]
    visited_states = np.concatenate([batch[0].cpu().numpy().reshape(-1) for batch in batches])
    visited_shift_up = np.concatenate([batch[2].cpu().numpy()[:, 0] for batch in batches])
    assert batch_sizes == [2, 2, 1]
    np.testing.assert_allclose(visited_states, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(visited_shift_up, [0, 1, 0, 1, 0])


def test_agent_training_state_preserves_steps_beyond_buffer_occupancy(tmp_path):
    src = Agent(
        DummyAgentEnv(),
        DummyAgentEnv(),
        DummyAlgo(),
        str(tmp_path / "src"),
        torch.device("cpu"),
        use_offline_buffer=True,
        memory_size=8,
        offline_buffer_size=8,
        batch_size=2,
        eval_interval=0,
    )
    dst = Agent(
        DummyAgentEnv(),
        DummyAgentEnv(),
        DummyAlgo(),
        str(tmp_path / "dst"),
        torch.device("cpu"),
        use_offline_buffer=True,
        memory_size=8,
        offline_buffer_size=8,
        batch_size=2,
        eval_interval=0,
    )
    try:
        for index in range(3):
            append_agent_transition(src._replay_buffer, index)
        src._replay_buffer.online(True)
        for index in range(3, 5):
            append_agent_transition(src._replay_buffer, index)

        src._steps = 12345
        src._episodes = 67
        src._demo_transition_count = 89
        src.best_lap_time = 72.5
        src.best_reward = 100.25
        src._best_eval_score = 12.5

        checkpoint_path = tmp_path / "checkpoint"
        src.save(str(checkpoint_path), save_buffer=True)
        dst.load(str(checkpoint_path), load_buffer=True)

        assert dst._steps == 12345
        assert dst._episodes == 67
        assert dst._demo_transition_count == 89
        assert dst.best_lap_time == 72.5
        assert dst.best_reward == 100.25
        assert dst._best_eval_score == 12.5
        assert dst._replay_buffer.offline_size == 3
        assert dst._replay_buffer.online_size == 2
        np.testing.assert_allclose(
            dst._replay_buffer._shift_labels[:2],
            src._replay_buffer._shift_labels[:2],
        )
        np.testing.assert_allclose(
            dst._replay_buffer._offline._shift_labels[:3],
            src._replay_buffer._offline._shift_labels[:3],
        )
    finally:
        src._writer.close()
        dst._writer.close()


def test_agent_training_state_saved_when_checkpoint_skips_replay_buffer(tmp_path):
    src = Agent(
        DummyAgentEnv(),
        DummyAgentEnv(),
        DummyAlgo(),
        str(tmp_path / "src"),
        torch.device("cpu"),
        use_offline_buffer=True,
        memory_size=8,
        offline_buffer_size=8,
        batch_size=2,
        eval_interval=0,
    )
    dst = Agent(
        DummyAgentEnv(),
        DummyAgentEnv(),
        DummyAlgo(),
        str(tmp_path / "dst"),
        torch.device("cpu"),
        use_offline_buffer=True,
        memory_size=8,
        offline_buffer_size=8,
        batch_size=2,
        eval_interval=0,
    )
    try:
        src._steps = 2468
        src._episodes = 24

        checkpoint_path = tmp_path / "checkpoint_without_buffer"
        src.save(str(checkpoint_path), save_buffer=False)

        assert (checkpoint_path / TRAINING_STATE_FILENAME).exists()
        assert not (checkpoint_path / "replay_buffer.pkl").exists()

        dst.load(str(checkpoint_path), load_buffer=True)

        assert dst._steps == 2468
        assert dst._episodes == 24
        assert len(dst._replay_buffer) == 0
    finally:
        src._writer.close()
        dst._writer.close()


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


def test_reward_ignores_neutral_reverse_and_out_of_track_penalties():
    env = make_env_stub()
    state = {"speed": 30.0, "gap": 0.0, "actualGear": 0, "out_of_track": 1.0}

    reward = env.get_reward(state, np.zeros(3, dtype=np.float32)).item()

    assert np.isclose(reward, 0.36)
    assert "gear_shift_reward" not in state
    assert "out_of_track_penalty" not in state


def test_autoshifter_upshifts_from_high_rpm_with_throttle():
    shifter = AutoShifter({"max_rpm": 8000.0, "idle_rpm": 1000.0}, ctrl_rate=25)

    action, info = shifter.update(
        {"actualGear": 3, "accStatus": 1.0, "brakeStatus": 0.0, "RPM": 7900.0, "speed": 40.0},
        dt=0.04,
    )

    np.testing.assert_allclose(action, [1.0, 0.0])
    assert info["auto_aggressiveness"] > 0.0


def test_autoshifter_recovers_neutral_to_first_at_idle():
    shifter = AutoShifter({"max_rpm": 8000.0, "idle_rpm": 1000.0}, ctrl_rate=25)

    action, info = shifter.update(
        {"actualGear": 0, "accStatus": 0.0, "brakeStatus": 0.0, "RPM": 1000.0, "speed": 0.0},
        dt=0.04,
    )

    np.testing.assert_allclose(action, [1.0, 0.0])
    assert info["auto_gear"] == 0


def test_autoshifter_holds_first_at_idle():
    shifter = AutoShifter({"max_rpm": 8000.0, "idle_rpm": 1000.0}, ctrl_rate=25)

    action, _ = shifter.update(
        {"actualGear": 1, "accStatus": 0.0, "brakeStatus": 0.0, "RPM": 1000.0, "speed": 0.0},
        dt=0.04,
    )

    np.testing.assert_allclose(action, [0.0, 0.0])


def test_autoshifter_uses_gear_offset_for_neutral_and_first():
    neutral_shifter = AutoShifter(
        {"max_rpm": 8000.0, "idle_rpm": 1000.0, "gear_index_offset": 1},
        ctrl_rate=25,
    )

    action, info = neutral_shifter.update(
        {"actualGear": 1, "accStatus": 0.0, "brakeStatus": 0.0, "RPM": 1000.0, "speed": 0.0},
        dt=0.04,
    )
    np.testing.assert_allclose(action, [1.0, 0.0])
    assert info["auto_gear"] == 0

    first_shifter = AutoShifter(
        {"max_rpm": 8000.0, "idle_rpm": 1000.0, "gear_index_offset": 1},
        ctrl_rate=25,
    )

    action, info = first_shifter.update(
        {"actualGear": 2, "accStatus": 0.0, "brakeStatus": 0.0, "RPM": 1000.0, "speed": 0.0},
        dt=0.04,
    )
    np.testing.assert_allclose(action, [0.0, 0.0])
    assert info["auto_gear"] == 1


def test_autoshifter_config_can_wait_for_high_upshift_rpm():
    shifter = AutoShifter(
        {
            "max_rpm": 8000.0,
            "idle_rpm": 1000.0,
            "mode": "sport",
            "max_shift_rpm_ratio": 0.99,
            "rpm_range_divisor": 1.5,
        },
        ctrl_rate=25,
    )

    action, _ = shifter.update(
        {"actualGear": 3, "accStatus": 1.0, "brakeStatus": 0.0, "RPM": 7800.0, "speed": 40.0},
        dt=0.04,
    )
    np.testing.assert_allclose(action, [0.0, 0.0])

    action, _ = shifter.update(
        {"actualGear": 3, "accStatus": 1.0, "brakeStatus": 0.0, "RPM": 7950.0, "speed": 40.0},
        dt=0.04,
    )
    np.testing.assert_allclose(action, [1.0, 0.0])


def test_autoshifter_config_waits_for_low_downshift_rpm():
    shifter = AutoShifter(
        {
            "max_rpm": 8000.0,
            "idle_rpm": 1000.0,
            "mode": "sport",
            "max_shift_rpm_ratio": 0.99,
            "rpm_range_divisor": 1.5,
        },
        ctrl_rate=25,
    )

    action, _ = shifter.update(
        {"actualGear": 4, "accStatus": 0.0, "brakeStatus": 0.0, "RPM": 2500.0, "speed": 20.0},
        dt=0.04,
    )
    np.testing.assert_allclose(action, [0.0, 0.0])

    action, _ = shifter.update(
        {"actualGear": 4, "accStatus": 0.0, "brakeStatus": 0.0, "RPM": 1200.0, "speed": 20.0},
        dt=0.04,
    )
    np.testing.assert_allclose(action, [0.0, 1.0])


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
