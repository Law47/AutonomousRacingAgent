"""
Real-time Autonomous Driver for Assetto Corsa
Reads telemetry via dataReciever server → Feeds to trained model → Outputs to virtual controller
"""

import sys
import os
import time
import keyboard
import numpy as np
import pickle
import torch
import threading

# Import telemetry server and controller
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AgentBrain"))
from dataReciever import server_program, get_latest_telemetry
from input import AssettoController


class ModelDrivenController:
    """
    Autonomous driver that reads AC telemetry from dataReciever, runs model inference, and controls the car
    """
    
    def __init__(self):
        print("="*60)
        print("Initializing Autonomous Driver (Model-Driven Controller)")
        print("="*60)
        
        # Start telemetry server in background thread
        self.server_thread = threading.Thread(target=server_program, daemon=True)
        self.server_thread.start()
        print("✓ Telemetry server started (waiting for AC connections)")
        time.sleep(1)  # Give server time to bind
        
        # Initialize virtual controller
        self.controller = AssettoController()
        print("✓ Virtual controller initialized")
        
        # Load trained model
        agent_brain_path = os.path.join(os.path.dirname(__file__), "..", "AgentBrain")
        model_path = os.path.join(agent_brain_path, "model.pth")
        
        self.model = self._load_model(model_path)
        print("✓ Model loaded")
        
        # Load scalers
        scaler_X_path = os.path.join(agent_brain_path, "scaler_X.pkl")
        scaler_y_path = os.path.join(agent_brain_path, "scaler_y.pkl")
        
        with open(scaler_X_path, "rb") as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f:
            self.scaler_y = pickle.load(f)
        print("✓ Scalers loaded")
        
        self.running = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"✓ Using {self.device} device for inference")
        print("="*60)
        print("Ready to drive!")
        print("  1. Make sure Assetto Corsa is sending telemetry to localhost:8000")
        print("  2. Press [ to activate autonomous mode")
        print("  3. Press ] to deactivate and exit")
        print("="*60 + "\n")
    
    @staticmethod
    def _load_model(model_path):
        """Load trained neural network model"""
        class NeuralNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = torch.nn.Flatten()
                self.linear_relu_stack = torch.nn.Sequential(
                    torch.nn.Linear(105, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 3)
                )
            
            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits
        
        model = NeuralNetwork()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        # Optimize with TorchScript
        try:
            model = torch.jit.trace(model, torch.randn(1, 105, device=device))
        except Exception:
            pass  # If tracing fails, use unoptimized model
        
        return model
    
    def get_telemetry(self):
        """
        Get latest telemetry from dataReciever server
        Returns: numpy array of 105 features (float32) or None if no data
        """
        X, y = get_latest_telemetry()
        
        if X is None or len(X) == 0:
            return None
        
        # Get the last row (most recent packet)
        features = X[-1].astype(np.float32)
        
        if len(features) == 105:
            return features
        else:
            return None
    
    def predict_controls(self, telemetry_features):
        """
        Run model inference on telemetry features
        
        Args:
            telemetry_features: numpy array of 105 features
        
        Returns:
            dict with 'throttle', 'brake', 'steering' (all clamped to valid ranges)
        """
        try:
            # Normalize input
            telemetry_normalized = self.scaler_X.transform([telemetry_features])[0]
            
            # Convert to tensor
            input_tensor = torch.from_numpy(telemetry_normalized).to(torch.float32).to(self.device)
            input_tensor = input_tensor.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Denormalize output
            output_np = output.cpu().numpy()
            output_denormalized = self.scaler_y.inverse_transform(output_np)[0]
            
            # Clamp to valid ranges
            return {
                'throttle': float(np.clip(output_denormalized[0], 0.0, 1.0)),
                'brake': float(np.clip(output_denormalized[1], 0.0, 1.0)),
                'steering': float(np.clip(output_denormalized[2], -1.0, 1.0))
            }
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def run(self):
        """Main control loop"""
        print("Waiting for [ to start autonomous mode...\n")
        
        try:
            while True:
                # Check for exit
                if keyboard.is_pressed(']'):
                    print("\nExiting...")
                    if self.running:
                        self.stop_driving()
                    self.controller.reset()
                    break
                
                # Check for activation
                if keyboard.is_pressed('['):
                    if not self.running:
                        print("[AUTONOMOUS MODE ACTIVATED]")
                        self.running = True
                        self.controller.reset()
                
                # Main driving loop
                if self.running:
                    telemetry = self.get_telemetry()
                    
                    if telemetry is not None and len(telemetry) == 105:
                        controls = self.predict_controls(telemetry)
                        
                        if controls is not None:
                            self.controller.send_inputs(
                                steering=controls['steering'],
                                throttle=controls['throttle'],
                                brake=controls['brake']
                            )
                        
                        time.sleep(0.01)  # ~100 Hz update rate
                    else:
                        time.sleep(0.05)
                else:
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nInterrupted")
            self.stop_driving()
            self.controller.reset()
        except Exception as e:
            print(f"Fatal error: {e}")
            self.stop_driving()
            self.controller.reset()
            raise
    
    def stop_driving(self):
        """Deactivate autonomous mode"""
        self.running = False
        self.controller.reset()
        print("[AUTONOMOUS MODE DEACTIVATED]")


if __name__ == "__main__":
    driver = ModelDrivenController()
    driver.run()
