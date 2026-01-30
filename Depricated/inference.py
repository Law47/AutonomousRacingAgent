import torch
import torch.jit
import numpy as np
import os
import pickle

class RacingAgentInference:
    
    def __init__(self, model_path="model.pth"):
        # Determine device (GPU for speed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Inference running on {self.device}")
        
        # Get absolute path to model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(script_dir, model_path)
        
        # Load model state dict
        state_dict = torch.load(full_model_path, map_location=self.device, weights_only=True)
        
        # Define model architecture (must match training)
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Optimize with TorchScript for faster inference
        # This compiles the model to optimized bytecode
        self.model = torch.jit.trace(
            self.model,
            torch.randn(1, 105, device=self.device)
        )
        
        # Load scalers for data normalization
        scaler_X_path = os.path.join(script_dir, "scaler_X.pkl")
        scaler_y_path = os.path.join(script_dir, "scaler_y.pkl")
        
        with open(scaler_X_path, "rb") as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f:
            self.scaler_y = pickle.load(f)
        
        print("Model loaded and optimized for inference")
    
    @staticmethod
    def _build_model():
        """Build the neural network model"""
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
        
        return NeuralNetwork()
    
    @torch.inference_mode()  # Faster than no_grad() + eval()
    def predict(self, telemetry_input):
        """
        Predict steering, throttle, and brake values from telemetry.
        
        Args:
            telemetry_input: numpy array or list of 105 features
        
        Returns:
            dict with keys: 'throttle', 'brake', 'steering'
        """
        # Convert input to numpy array if needed
        if not isinstance(telemetry_input, np.ndarray):
            telemetry_input = np.array(telemetry_input, dtype=np.float32)
        
        # Normalize input using the training scaler
        if telemetry_input.ndim == 1:
            telemetry_normalized = self.scaler_X.transform([telemetry_input])[0]
        else:
            telemetry_normalized = self.scaler_X.transform(telemetry_input)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(telemetry_normalized).to(torch.float32)
        
        # Add batch dimension if needed
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Denormalize output
        output_np = output[0].cpu().numpy().reshape(1, -1)
        output_denormalized = self.scaler_y.inverse_transform(output_np)[0]
        
        return {
            'throttle': float(output_denormalized[0]),
            'brake': float(output_denormalized[1]),
            'steering': float(output_denormalized[2])
        }
    
    def predict_batch(self, telemetry_inputs):
        """
        Predict for multiple inputs at once (more efficient for batch processing).
        
        Args:
            telemetry_inputs: numpy array of shape (batch_size, 105)
        
        Returns:
            numpy array of shape (batch_size, 3) with [throttle, brake, steering] for each input
        """
        if isinstance(telemetry_inputs, list):
            telemetry_inputs = np.array(telemetry_inputs, dtype=np.float32)
        
        # Normalize inputs
        telemetry_normalized = self.scaler_X.transform(telemetry_inputs)
        
        input_tensor = torch.from_numpy(telemetry_normalized).to(torch.float32).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Denormalize outputs
        output_np = output.cpu().numpy()
        output_denormalized = self.scaler_y.inverse_transform(output_np)
        
        return output_denormalized


# Example usage
if __name__ == "__main__":
    # Initialize inference engine
    agent = RacingAgentInference("model.pth")
    
    # Example: Create dummy telemetry input (105 features)
    dummy_telemetry = np.random.randn(105).astype(np.float32)
    
    # Single prediction
    print("\n--- Single Prediction ---")
    result = agent.predict(dummy_telemetry)
    print(f"Throttle: {result['throttle']:.4f}")
    print(f"Brake: {result['brake']:.4f}")
    print(f"Steering: {result['steering']:.4f}")
    
    # Batch prediction (more efficient)
    print("\n--- Batch Prediction (10 samples) ---")
    batch_telemetry = np.random.randn(10, 105).astype(np.float32)
    batch_results = agent.predict_batch(batch_telemetry)
    print(f"Output shape: {batch_results.shape}")
    print(f"First sample - Throttle: {batch_results[0, 0]:.4f}, Brake: {batch_results[0, 1]:.4f}, Steering: {batch_results[0, 2]:.4f}")
    
    # Benchmark latency
    print("\n--- Latency Benchmark ---")
    import time
    
    num_iterations = 1000
    single_input = np.random.randn(105).astype(np.float32)
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        agent.predict(single_input)
    elapsed = time.perf_counter() - start
    
    avg_latency_ms = (elapsed / num_iterations) * 1000
    print(f"Average latency per prediction: {avg_latency_ms:.3f}ms")
    print(f"Predictions per second: {1000/avg_latency_ms:.0f}")
