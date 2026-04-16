import numpy as np


DEFAULT_SHIFT_LABEL_MIN_DRIVE_GEAR = 2


def get_state_gear(state, default=0):
    return int(state.get("actualGear", default))


def is_stable_drive_gear(gear, min_drive_gear=DEFAULT_SHIFT_LABEL_MIN_DRIVE_GEAR):
    return int(gear) >= int(min_drive_gear)


def infer_shift_from_stable_gear(
    current_gear,
    previous_stable_gear,
    min_drive_gear=DEFAULT_SHIFT_LABEL_MIN_DRIVE_GEAR,
):
    """Infer one model shift pulse from stable drive-gear transitions.

    Assetto Corsa's raw gear channel commonly reports reverse/neutral/shift
    transitions before the next drive gear arrives. Ignoring those unstable
    values prevents labels like 6 -> 1 -> 7 from becoming downshift+upshift.
    """
    shift_actions = np.zeros(2, dtype=np.float32)
    current_gear = int(current_gear)

    if not is_stable_drive_gear(current_gear, min_drive_gear=min_drive_gear):
        return shift_actions, previous_stable_gear, False

    if previous_stable_gear is None:
        return shift_actions, current_gear, True

    gear_delta = current_gear - int(previous_stable_gear)
    if gear_delta > 0:
        shift_actions[0] = 1.0
    elif gear_delta < 0:
        shift_actions[1] = 1.0

    return shift_actions, current_gear, True


def infer_shift_from_state(
    current_state,
    previous_stable_gear,
    min_drive_gear=DEFAULT_SHIFT_LABEL_MIN_DRIVE_GEAR,
):
    current_gear = get_state_gear(current_state)
    return infer_shift_from_stable_gear(
        current_gear,
        previous_stable_gear,
        min_drive_gear=min_drive_gear,
    )
