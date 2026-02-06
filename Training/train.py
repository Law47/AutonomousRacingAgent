from pathlib import Path
import sys, os
import gymnasium as gym

trainingPath = Path(__file__).parent.parent
modelPath = Path(__file__).parent.parent / 'Model'
discorPath = modelPath / 'discor'
sys.path.extend([str(trainingPath), str(modelPath), str(discorPath)])

from discor.algorithm import SAC, DisCor
from discor.agent import Agent

