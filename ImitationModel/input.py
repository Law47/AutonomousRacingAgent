import vgamepad as vg
import time
import keyboard

class AssettoController:
    def __init__(self):
        # Create virtual Xbox 360 controller
        self.gamepad = vg.VX360Gamepad()
        print("Virtual controller created. Make sure Assetto Corsa recognizes it in settings.")
        # Ensure controller starts in a neutral state
        # Steering center (0.0), triggers unpressed (0.0)
        try:
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.gamepad.right_trigger_float(value_float=0.0)
            self.gamepad.left_trigger_float(value_float=0.0)
            self.gamepad.update()
        except Exception:
            # If the backend doesn't support direct float-setting yet, fall back to reset
            try:
                self.gamepad.reset()
                self.gamepad.update()
            except Exception:
                pass
    
    def send_inputs(self, steering, throttle, brake):
        """
        Send driving inputs to Assetto Corsa
        
        Args:
            steering: float, -1.0 (full left) to 1.0 (full right)
            throttle: float, 0.0 to 1.0
            brake: float, 0.0 to 1.0
        """
        # Steering: left stick X axis
        self.gamepad.left_joystick_float(x_value_float=steering, y_value_float=0.0)
        
        # Throttle: right trigger (0.0 to 1.0)
        self.gamepad.right_trigger_float(value_float=throttle)
        
        # Brake: left trigger (0.0 to 1.0)
        self.gamepad.left_trigger_float(value_float=brake)
        
        # Submit the input state
        self.gamepad.update()
    
    def press_button(self, button):
        """Press a button (for shifting, etc.)"""
        self.gamepad.press_button(button)
        self.gamepad.update()
    
    def release_button(self, button):
        """Release a button"""
        self.gamepad.release_button(button)
        self.gamepad.update()
    
    def reset(self):
        """Reset all inputs to neutral"""
        # Explicitly set neutral positions to avoid default 50% throttle/clutch
        try:
            # Center steering axis
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            # Ensure triggers are unpressed (0.0 = released)
            self.gamepad.right_trigger_float(value_float=0.0)
            self.gamepad.left_trigger_float(value_float=0.0)
            # Also reset buttons/state as a fallback
            self.gamepad.reset()
            self.gamepad.update()
        except Exception:
            # Best-effort fallback
            try:
                self.gamepad.reset()
                self.gamepad.update()
            except Exception:
                pass
    
    def __del__(self):
        # Clean up
        self.reset()


# Example usage
if __name__ == "__main__":
    controller = AssettoController()
    
    print("Controller initialized!")
    print("Press [ to run test inputs")
    print("Press ] to exit the program")
    print()
    
    while True:
        # Wait for '[' key
        if keyboard.is_pressed('['):
            print("Running test inputs...")
            
            # Test: gentle right turn with half throttle
            controller.send_inputs(steering=0.3, throttle=0.5, brake=0.0)
            time.sleep(2)
            
            # Full brake
            controller.send_inputs(steering=0.0, throttle=0.0, brake=1.0)
            time.sleep(1)
            
            # Reset to neutral
            controller.reset()
            print("Test complete! Press [ again or ] to exit")
            
            # Wait a bit to avoid rapid re-triggering
            time.sleep(0.5)
        
        # Check for ']' key to exit
        if keyboard.is_pressed(']'):
            print("Exiting...")
            controller.reset()
            break
        
        # Small sleep to prevent excessive CPU usage
        time.sleep(0.01)