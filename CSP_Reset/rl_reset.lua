--[[
  RL Reset - CSP Lua Script for Assetto Corsa
  
  PURPOSE: Resets the player car to the start position when triggered
  by the Python RL training script via a flag file.
  
  INSTALLATION:
    1. Open Content Manager
    2. Go to Settings -> Custom Shaders Patch -> Lua Apps
       (or in newer CM: Content -> Lua Apps)
    3. Click "Add new script" or "+" button
    4. Paste this entire script, or point it to this file
    5. Make sure the script is ENABLED
    6. Start a session - the script will monitor for reset commands
    
  ALTERNATIVE INSTALLATION (manual):
    1. Copy this file to:
       <AC_DIR>/extension/lua/joypad-assist/rl_reset.lua
       OR
       <Documents>/Assetto Corsa/cfg/extension/lua/online-scripts/rl_reset.lua
    2. Enable it in Content Manager settings
    
  HOW IT WORKS:
    - Checks every 100ms for a flag file at: %USERPROFILE%/ac_reset_flag.txt
    - When found, teleports car to pit/start position
    - Writes confirmation to: %USERPROFILE%/ac_reset_response.txt
    - Deletes the flag file
]]--

local HOME = os.getenv("USERPROFILE") or os.getenv("HOME") or "C:/Users/Default"
local FLAG_FILE = HOME .. "/ac_reset_flag.txt"
local RESPONSE_FILE = HOME .. "/ac_reset_response.txt"
local CHECK_INTERVAL = 0.1  -- seconds between file checks

local timer = 0

-- Helper: check if file exists
local function fileExists(path)
  local f = io.open(path, "r")
  if f then
    f:close()
    return true
  end
  return false
end

-- Helper: read file contents
local function readFile(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local content = f:read("*all")
  f:close()
  return content
end

-- Helper: write file
local function writeFile(path, content)
  local f = io.open(path, "w")
  if not f then return false end
  f:write(content)
  f:close()
  return true
end

-- Helper: delete file
local function deleteFile(path)
  os.remove(path)
end

-- Reset the car
local function resetCar()
  -- Use physics.setCarPosition to teleport to start
  -- physics namespace is available in CSP Lua context
  local car = ac.getCar(0)  -- player car is index 0
  
  if car then
    -- Method 1: Teleport to pit position (most reliable)
    physics.teleportCarTo(0, ac.SpawnSet.Pits)
    
    -- Zero out velocity
    physics.setCarVelocity(0, vec3(0, 0, 0))
    physics.setCarAngularVelocity(0, vec3(0, 0, 0))
    
    ac.log("[RL_Reset] Car reset to pits via teleport")
    return true
  end
  
  ac.log("[RL_Reset] Could not get car reference")
  return false
end

-- Main update function called every frame
function script.update(dt)
  timer = timer + dt
  
  if timer < CHECK_INTERVAL then
    return
  end
  timer = 0
  
  -- Check for reset flag
  if not fileExists(FLAG_FILE) then
    return
  end
  
  local content = readFile(FLAG_FILE)
  if not content then return end
  
  ac.log("[RL_Reset] Reset command received: " .. tostring(content))
  
  -- Delete flag file immediately
  deleteFile(FLAG_FILE)
  
  -- Perform the reset
  local success = resetCar()
  
  -- Write response
  if success then
    writeFile(RESPONSE_FILE, "ok")
    ac.log("[RL_Reset] Reset complete, confirmation written")
  else
    writeFile(RESPONSE_FILE, "failed")
    ac.log("[RL_Reset] Reset failed")
  end
end
