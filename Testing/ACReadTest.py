import ctypes
import mmap
import time

# -----------------------------
# Assetto Corsa Physics Struct
# -----------------------------
class ACPhysics(ctypes.Structure):
    _fields_ = [
        ("packetId", ctypes.c_int),
        ("gas", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("fuel", ctypes.c_float),
        ("gear", ctypes.c_int),
        ("rpms", ctypes.c_float),
        ("steerAngle", ctypes.c_float),
        ("speedKmh", ctypes.c_float),
        ("velocity", ctypes.c_float * 3),
        ("accG", ctypes.c_float * 3),
        ("wheelSlip", ctypes.c_float * 4),
        ("wheelLoad", ctypes.c_float * 4),
    ]


# -----------------------------
# Open Shared Memory
# -----------------------------
SHM_NAME = "acpmf_physics"
SHM_SIZE = ctypes.sizeof(ACPhysics)

try:
    mm = mmap.mmap(
        -1,
        SHM_SIZE,
        tagname=SHM_NAME,
        access=mmap.ACCESS_READ
    )
except FileNotFoundError:
    print("ERROR: Assetto Corsa is not running or shared memory not available.")
    exit(1)

print("Connected to Assetto Corsa shared memory.")
print("Reading telemetry...\n")

# -----------------------------
# Read Loop (sync to physics)
# -----------------------------
last_packet = -1

while True:
    # Read fresh data from shared memory each iteration
    physics = ACPhysics.from_buffer_copy(mm)
    
    if physics.packetId != last_packet:
        last_packet = physics.packetId

        print(
            f"Speed: {physics.speedKmh:6.1f} km/h | "
            f"Steer: {physics.steerAngle:+.3f} rad | "
            f"Throttle: {physics.gas:.2f} | "
            f"Brake: {physics.brake:.2f} | "
            f"RPM: {physics.rpms:6.0f} | "
            f"Gear: {physics.gear}"
        )

    time.sleep(0.001)  # ~1000 Hz poll (cheap)