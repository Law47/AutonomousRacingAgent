## AUTONOMOUS RACING AGENT

An imitation-learning–based autonomous driving agent designed to race in Assetto Corsa.
The agent observes state information from the simulator and outputs steering, throttle,
and brake commands, which are injected into the game in real time.


#### PROJECT GOALS

- Train an autonomous racing agent using imitation and reinforcement learning
- Achieve minimal input latency for real-time driving control
- Interface directly with Assetto Corsa using virtual gamepad inputs
- Build a modular pipeline for:
  - Data collection
  - Model training
  - Inference and control


NOTE: The project's model.pth files use CUDA so please install with the CUDA version of pytorch

#### DEPENDENCIES

##### Core:
- Python 3.13.2
- Numpy 2.1.0
- Pytorch Build 2.10.0
- Cuda 12.6
- scikit-learn 1.8.0
- Pandas 3.0.0

##### Game Input / Control:
- ViGEmBus (REQUIRED)
  Virtual Gamepad Emulation Bus used to emulate an Xbox 360 controller with extremely low latency
  https://github.com/nefarius/ViGEmBus/releases
- vgamepad (Python interface for sending inputs through ViGEmBus)
- For training, bind Ctrl + R in content manager to restart race

##### Simulator:
- Assetto Corsa (PC)


### INSTALLATION

#### 1. Install ViGEmBus (Required)

ViGEmBus must be installed system-wide before running the agent.

Steps:
- Download the installer from 
##### https://github.com/nefarius/ViGEmBus/releases
- Install and reboot your system


#### 2. Clone the Repository

Command:
- ##### `git clone https://github.com/yourusername/autonomousracingagent.git`
- `cd autonomousracingagent`


#### 3. Install Dependencies

Command:
- ##### `conda env create -f .\p309.yml`

- If your machine has a cuda-compatible GPU then it is highly recommended to override the torch and torch vision installation in the conda environment. 

##### `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`