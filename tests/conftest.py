import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'Model')
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
