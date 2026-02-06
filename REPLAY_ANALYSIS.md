# Assetto Corsa Replay and Timing: The Real Problem

## Why Your Input Replay Has a -3ms Timing Drift

### The Fundamental Issue

You've been trying to **replay inputs** (steering, throttle, brake) through a virtual Xbox controller to reproduce a recorded lap. The problem is:

1. **AC stores PHYSICAL DATA in replays, not inputs**
   - Position (x, y, z)
   - Velocity vectors
   - Rotation/orientation
   - Suspension state
   - Engine state

2. **You can't perfectly recreate physics by replaying inputs**
   - Physics engines are sensitive to initial conditions and timing
   - Even tiny timing errors (3ms) compound over a lap
   - Different input processing latencies create small divergences
   - These divergences accumulate, causing the car to turn too early/late

3. **Your -3ms drift is not fixable with timing compensation**
   - The drift is a symptom of fundamental physics divergence
   - Compensation masks the real problem temporarily
   - Different sections of the track will have different timing issues

### Why the Official AC Replay Works Perfectly

Assetto Corsa's built-in replay doesn't replay inputs at all:
- It **stores interpolated snapshots** of physical state at regular intervals
- During playback, it **smoothly interpolates between snapshots**
- The physics engine is **NOT re-run** during replay
- Result: Perfect reproduction with zero drift

## The Solution: Use Actual Replay Data

### Option 1: Extract Data from .rpy Files (Recommended)

Use `acreplay-parser` to convert `.rpy` files to CSV:

```bash
# Install the tool
pip install acreplay-parser

# Convert a replay
python ACReplayFromRpy.py <path-to-replay.rpy>

# Or batch convert
python ACReplayFromRpy.py --batch <directory>
```

**Advantages:**
- ✅ Get exact physical data AC recorded
- ✅ Zero timing drift
- ✅ Perfect for training your autonomous agent
- ✅ Use actual car positions/velocities, not estimated data

### Option 2: Keep Your Current Recording (Still Useful)

Your current CSV recording with physical telemetry + controller inputs is still valuable:
- Use `steerangle_rad`, `gas`, `brake` from the CSV (these are physics data)
- Ignore the timing drift - it's acceptable for training data
- Your controller input columns are useful for correlating inputs to outputs

## Why Your Project Should Use Replay Files

For your AutonomousRacingAgent:
1. Replays contain ground-truth vehicle physics data
2. Much cleaner data than trying to re-simulate physics with input replay
3. `acreplay-parser` is well-maintained and handles AC's binary format
4. You get frame-interpolated smooth trajectories (better for ML training)

## Recommendation

1. **For data collection**: Use your current ACRead.py to record driving sessions
2. **For training data**: Convert those sessions using ACReplayFromRpy.py
3. **For replay verification**: Use AC's built-in replay system (it's perfect)
4. **For input capture**: Your controller input recording is correct - use it for analysis but don't try to replay inputs

This approach:
- Eliminates the -3ms drift problem entirely
- Gives you high-quality physics-based training data
- Scales to hundreds of replays for training
- Matches how professional racing sims handle telemetry

## Technical Details: Why Timing Drift Exists

Your current approach has this loop:
```
Record frame N: physics_state[N], controller_input[N] at time T_N
Replay frame N: apply controller_input[N] → physics engine
               → new physics_state'[N] at time T'_N
```

The issue:
- `physics_state[N]` ≠ `physics_state'[N]` due to:
  - Initial state differences
  - Controller input processing latency
  - AC's physics update ordering
  - Floating-point precision differences
  
Result: Over 13,000 frames, these small errors accumulate into noticeable timing differences.

The solution used by AC:
```
Record frame N: physics_state[N] (position, velocity, etc.)
Replay frame N: interpolate(physics_state[N], physics_state[N+1])
```

No physics engine involved = perfect fidelity.
