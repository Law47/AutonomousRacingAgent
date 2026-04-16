import ctypes
import os
import shutil
import struct, time
import math
from pathlib import Path

DLL_NAME = "vJoyInterface.dll"
CONST_DLL_VJOY = "C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll"
VJOY_DLL_ENV_VARS = ("VJOY_DLL_PATH", "VJOYINTERFACE_DLL", "VJOY_PATH")
_DLL_DIRECTORY_HANDLES = []


def _candidate_vjoy_dll_paths():
    candidates = []

    for env_var in VJOY_DLL_ENV_VARS:
        env_value = os.environ.get(env_var)
        if not env_value:
            continue
        env_path = Path(env_value)
        candidates.append(env_path / DLL_NAME if env_path.is_dir() else env_path)

    program_roots = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        "C:\\Program Files",
        "C:\\Program Files (x86)",
    ]
    relative_locations = [
        ("vJoy", "x64", DLL_NAME),
        ("vJoy", "x86", DLL_NAME),
        ("vJoy", "SDK", "lib", "amd64", DLL_NAME),
        ("vJoy", "SDK", "lib", "x86", DLL_NAME),
    ]
    for root in program_roots:
        if not root:
            continue
        for relative_location in relative_locations:
            candidates.append(Path(root).joinpath(*relative_location))

    path_match = shutil.which(DLL_NAME)
    if path_match:
        candidates.append(Path(path_match))

    candidates.append(Path(CONST_DLL_VJOY))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        normalized = str(candidate)
        normalized_key = normalized.lower()
        if normalized_key not in seen:
            seen.add(normalized_key)
            unique_candidates.append(normalized)
    return unique_candidates


def resolve_vjoy_dll_path():
    for candidate in _candidate_vjoy_dll_paths():
        if os.path.exists(candidate):
            return candidate

    searched = "\n  - ".join(_candidate_vjoy_dll_paths())
    raise FileNotFoundError(
        f"Unable to find {DLL_NAME}. Install vJoy or set VJOY_DLL_PATH to the full DLL path. "
        f"Searched:\n  - {searched}"
    )


def load_vjoy_dll():
    dll_path = resolve_vjoy_dll_path()
    dll_dir = os.path.dirname(dll_path)
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_dir))
    return ctypes.CDLL(dll_path)

class vJoy(object):
    def __init__(self, reference=1):
        self.handle = None
        self.dll = load_vjoy_dll()
        self.reference = reference
        self.acquired = False

    def open(self):
        if self.dll.AcquireVJD(self.reference):
            self.acquired = True
            return True
        return False

    def close(self):
        if self.dll.RelinquishVJD(self.reference):
            self.acquired = False
            return True
        return False

    def generateJoystickPosition(self,
                                 wThrottle=0, wRudder=0, wAileron=0,
                                 wAxisX=0, wAxisY=0, wAxisZ=0,
                                 wAxisXRot=0, wAxisYRot=0, wAxisZRot=0,
                                 wSlider=0, wDial=0, wWheel=0,
                                 wAxisVX=0, wAxisVY=0, wAxisVZ=0,
                                 wAxisVBRX=0, wAxisVBRY=0, wAxisVBRZ=0,
                                 lButtons=0, bHats=0, bHatsEx1=0, bHatsEx2=0, bHatsEx3=0):
        """
        typedef struct _JOYSTICK_POSITION
        {
            BYTE    bDevice; // Index of device. 1-based
            LONG    wThrottle;
            LONG    wRudder;
            LONG    wAileron;
            LONG    wAxisX;
            LONG    wAxisY;
            LONG    wAxisZ;
            LONG    wAxisXRot;
            LONG    wAxisYRot;
            LONG    wAxisZRot;
            LONG    wSlider;
            LONG    wDial;
            LONG    wWheel;
            LONG    wAxisVX;
            LONG    wAxisVY;
            LONG    wAxisVZ;
            LONG    wAxisVBRX;
            LONG    wAxisVBRY;
            LONG    wAxisVBRZ;
            LONG    lButtons;   // 32 buttons: 0x00000001 means button1 is pressed, 0x80000000 -> button32 is pressed
            DWORD   bHats;      // Lower 4 bits: HAT switch or 16-bit of continuous HAT switch
                        DWORD   bHatsEx1;   // 16-bit of continuous HAT switch
                        DWORD   bHatsEx2;   // 16-bit of continuous HAT switch
                        DWORD   bHatsEx3;   // 16-bit of continuous HAT switch
        } JOYSTICK_POSITION, *PJOYSTICK_POSITION;
        """
        joyPosFormat = "BlllllllllllllllllllIIII"
        pos = struct.pack(joyPosFormat, self.reference, wThrottle, wRudder,
                          wAileron, wAxisX, wAxisY, wAxisZ, wAxisXRot, wAxisYRot,
                          wAxisZRot, wSlider, wDial, wWheel, wAxisVX, wAxisVY, wAxisVZ,
                          wAxisVBRX, wAxisVBRY, wAxisVBRZ, lButtons, bHats, bHatsEx1, bHatsEx2, bHatsEx3)
        return pos

    def update(self, joystickPosition):
        if self.dll.UpdateVJD(self.reference, joystickPosition):
            return True
        return False

    # Not working, send buttons one by one
    def sendButtons(self, bState):
        joyPosition = self.generateJoystickPosition(lButtons=bState)
        return self.update(joyPosition)

    def setButton(self, index, state):
        if self.dll.SetBtn(state, self.reference, index):
            return True
        return False

# Only for testing
def gearUp():
    #press
    setJoy(1 ,0.3, 0, 0x00000001, 16384)

    #release
    setJoy(1 ,0.3, 0, 0, 16384)
