import ctypes
from ctypes import c_int32, c_float, c_wchar

# ==================== Enums ====================
class AC_STATUS(ctypes.c_int):
    """Simulation status"""
    OFF = 0
    REPLAY = 1
    LIVE = 2
    PAUSE = 3

class AC_SESSION_TYPE(ctypes.c_int):
    """Session type"""
    PRACTICE = 0
    QUALIFY = 1
    RACE = 2
    HOTLAP = 3
    TIME_ATTACK = 4
    DRIFT = 5
    DRAG = 6

class AC_FLAG_TYPE(ctypes.c_int):
    """Flag status"""
    NO_FLAG = 0
    BLUE = 1
    YELLOW = 2
    BLACK = 3
    WHITE = 4
    CHECKERED = 5
    PENALTY = 6

# ==================== Physics Structure ====================
class SPageFilePhysics(ctypes.Structure):
    """All physics data from Assetto Corsa shared memory"""
    _pack_ = 4
    _fields_ = [
        ('packetId', c_int32),              # Packet counter
        ('gas', c_float),                   # Throttle (0-1)
        ('brake', c_float),                 # Brake (0-1)
        ('fuel', c_float),                  # Fuel amount (liters)
        ('gear', c_int32),                  # Current gear (-1=R, 0=N, 1+=forward)
        ('rpms', c_int32),                  # Engine RPM
        ('steerAngle', c_float),            # Steering wheel angle (radians)
        ('speedKmh', c_float),              # Car speed (km/h)
        ('velocity', c_float * 3),          # Velocity vector [x, y, z] (m/s)
        ('accG', c_float * 3),              # Acceleration in G's [x, y, z]
        ('wheelSlip', c_float * 4),         # Wheel slip [FL, FR, RL, RR]
        ('wheelLoad', c_float * 4),         # Wheel load [FL, FR, RL, RR]
        ('wheelsPressure', c_float * 4),    # Tire pressure [FL, FR, RL, RR]
        ('wheelAngularSpeed', c_float * 4), # Wheel angular speed [FL, FR, RL, RR]
        ('tyreWear', c_float * 4),          # Tire wear [FL, FR, RL, RR] (0-1)
        ('tyreDirtyLevel', c_float * 4),    # Tire dirt level [FL, FR, RL, RR]
        ('tyreCoreTemperature', c_float * 4), # Tire core temp [FL, FR, RL, RR]
        ('camberRAD', c_float * 4),         # Camber angle [FL, FR, RL, RR] (radians)
        ('suspensionTravel', c_float * 4),  # Suspension travel [FL, FR, RL, RR]
        ('drs', c_float),                   # DRS activation status
        ('tc', c_float),                    # Traction control level
        ('heading', c_float),               # Heading angle (radians)
        ('pitch', c_float),                 # Pitch angle (radians)
        ('roll', c_float),                  # Roll angle (radians)
        ('cgHeight', c_float),              # Center of gravity height
        ('carDamage', c_float * 5),         # Car damage [front, rear, left, right, centre]
        ('numberOfTyresOut', c_int32),      # Number of tires outside track limits
        ('pitLimiterOn', c_int32),          # Pit limiter enabled
        ('abs', c_float),                   # ABS level
        ('kersCharge', c_float),            # KERS charge level
        ('kersInput', c_float),             # KERS input level
        ('autoShifterOn', c_int32),         # Auto shifter enabled
        ('rideHeight', c_float * 2),        # Ride height [front, rear]
        ('turboBoost', c_float),            # Turbo boost pressure
        ('ballast', c_float),               # Ballast weight
        ('airDensity', c_float),            # Air density
        ('airTemp', c_float),               # Air temperature (°C)
        ('roadTemp', c_float),              # Road temperature (°C)
        ('localAngularVel', c_float * 3),   # Local angular velocity [x, y, z]
        ('finalFF', c_float),               # Final force feedback
        ('performanceMeter', c_float),      # Performance meter value
        ('engineBrake', c_int32),           # Engine brake level
        ('ersRecoveryLevel', c_int32),      # ERS recovery level
        ('ersPowerLevel', c_int32),         # ERS power level
        ('ersHeatCharging', c_int32),       # ERS heat charging
        ('ersIsCharging', c_int32),         # ERS is charging
        ('kersCurrentKJ', c_float),         # KERS current energy (kJ)
        ('drsAvailable', c_int32),          # DRS available
        ('drsEnabled', c_int32),            # DRS enabled
        ('brakeTemp', c_float * 4),         # Brake temperature [FL, FR, RL, RR]
        ('clutch', c_float),                # Clutch engagement (0-1)
        ('tyreTempI', c_float * 4),         # Tire inside temperature [FL, FR, RL, RR]
        ('tyreTempM', c_float * 4),         # Tire middle temperature [FL, FR, RL, RR]
        ('tyreTempO', c_float * 4),         # Tire outside temperature [FL, FR, RL, RR]
        ('isAIControlled', c_int32),        # Car is AI controlled
        ('tyreContactPoint', c_float * 12), # Tire contact points [FL, FR, RL, RR] x3
        ('tyreContactNormal', c_float * 12), # Tire contact normals [FL, FR, RL, RR] x3
        ('tyreContactHeading', c_float * 12), # Tire contact headings [FL, FR, RL, RR] x3
    ]


# ==================== Graphics Structure ====================
class SPageFileGraphic(ctypes.Structure):
    """All graphics/session data from Assetto Corsa shared memory"""
    _pack_ = 4
    _fields_ = [
        ('packetId', c_int32),              # Packet counter
        ('status', c_int32),                # Simulation status (AC_STATUS: OFF=0, REPLAY=1, LIVE=2, PAUSE=3)
        ('session', c_int32),               # Session type (AC_SESSION_TYPE: PRACTICE=0, QUALIFY=1, RACE=2, etc)
        ('currentTime', c_wchar * 15),      # Current lap time (string)
        ('lastTime', c_wchar * 15),         # Last lap time (string)
        ('bestTime', c_wchar * 15),         # Best lap time (string)
        ('split', c_wchar * 15),            # Current split time (string)
        ('completedLaps', c_int32),         # Number of completed laps
        ('position', c_int32),              # Current position in race
        ('iCurrentTime', c_int32),          # Current time in milliseconds
        ('iLastTime', c_int32),             # Last lap time in milliseconds
        ('iBestTime', c_int32),             # Best lap time in milliseconds
        ('sessionTimeLeft', c_float),       # Session time remaining (seconds)
        ('distanceTraveled', c_float),      # Distance traveled in session (km)
        ('isInPit', c_int32),               # Car is in pit
        ('currentSectorIndex', c_int32),    # Current sector (0, 1, 2)
        ('lastSectorTime', c_int32),        # Last sector time in milliseconds
        ('numberOfLaps', c_int32),          # Total number of laps
        ('tyreCompound', c_wchar * 33),     # Tire compound name
        ('replayTimeMultiplier', c_float),  # Replay time multiplier
        ('normalizedCarPosition', c_float), # Normalized car position on track
        ('carCoordinates', c_float * 3),    # Car coordinates [x, y, z]
        ('penaltyTime', c_float),           # Penalty time (seconds)
        ('flag', c_int32),                  # Flag status (AC_FLAG_TYPE: NO_FLAG=0, BLUE=1, YELLOW=2, etc)
        ('idealLineOn', c_int32),           # Ideal line display enabled
    ]