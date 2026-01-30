import torch
from torch import nn
from acPhysics import ACPhysics
import ctypes
import mmap
import time

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def kmphToMph(kmph):
    return kmph * 0.621371

def physicsPacketToTensor(packet):
    v = packet.velocity
    a = packet.accG
    wS = packet.wheelSlip
    wL = packet.wheelLoad
    cords = packet.carCoordinates

    data = [packet.gear, kmphToMph(packet.speedKmh), packet.rpms, v[0], v[1], v[2], a[0], a[1], a[2], wS[0], wS[1], wS[2], wS[3], wL[0], wL[1], wL[2], wL[3], cords[0], cords[1], cords[2]]
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    
# Load Model
model = NeuralNetwork()
model.load_state_dict(torch.load("./Model/model.pth", weights_only=True))
model.eval()

# Start AC connection
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
last_packet = -1

while True:
    # Read fresh data from shared memory each iteration
    physics = ACPhysics.from_buffer_copy(mm)
    
    if physics.packetId != last_packet:
        last_packet = physics.packetId

        last_packet = physics.packetId

        X = physicsPacketToTensor(physics)
        y = model(X).flatten()

        # printing model outputs
        print(
            f"Gas: {y[0]:.2f} | "
            f"Brake: {y[1]:.2f} | "
            f"Steer: {y[2]:.2f}"
        )

        # print(
        #     f"Speed: {physics.speedKmh:6.1f} km/h | "
        #     f"Steer: {physics.steerAngle:+.3f} rad | "
        #     f"Throttle: {physics.gas:.2f} | "
        #     f"Brake: {physics.brake:.2f} | "
        #     f"RPM: {physics.rpms:.2f} | "
        #     f"Gear: {physics.gear} | "
        #     f"Cords: {physics.carCoordinates[0]:.2f}, {physics.carCoordinates[1]:.2f}, {physics.carCoordinates[2]:.2f}"
        # )

    time.sleep(0.001)  # ~1000 Hz poll (cheap)