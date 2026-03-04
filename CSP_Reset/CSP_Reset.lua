--[[
  RL Reset - CSP Lua Script for Assetto Corsa
  
  PURPOSE: Resets the player car to the start position when triggered
  by the Python RL training script via a toggle file.
  
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
    - Checks every 100ms for toggle file at: %USERPROFILE%/ac_reset_toggle.txt
    - When file contains "1", teleports car to pit/start position
    - Writes "0" back to toggle file to confirm reset
]]--

local HOME = os.getenv("USERPROFILE") or os.getenv("HOME") or "C:/Users/Default"
local TOGGLE_FILE = HOME .. "/ac_reset_toggle.txt"
local RACING_LINE_CACHE_FILE = HOME .. "/ac_racing_line_cache.bin"
local CHECK_INTERVAL = 0.1  -- seconds between file checks

local timer = 0

-- Racing line cache
local RACING_LINE_CACHE = {
  points = {},
  point_count = 0,
  last_track = nil,
  last_layout = nil
}

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

-- Helper: write file (atomic write for toggle)
local function writeFile(path, content)
  local f = io.open(path, "w")
  if not f then 
    ac.log("[RL_Reset] ERROR: Could not open file for writing: " .. path)
    return false 
  end
  
  local success = f:write(content)
  f:close()  -- Close immediately to ensure flush on Windows
  
  if not success then
    ac.log("[RL_Reset] ERROR: Failed to write to file: " .. path)
    return false
  end
  
  return true
end

-- Write racing line to cache file for Python to read
local function write_racing_line_cache()
  local f = io.open(RACING_LINE_CACHE_FILE, "wb")
  if not f then
    ac.log("[RL_Racing_Line] Failed to open cache file for writing")
    return false
  end
  
  -- Write point count (4 bytes, little-endian int32)
  f:write(string.pack("<i4", RACING_LINE_CACHE.point_count))
  
  -- Write all points (x, y, z as 3 floats each)
  for i = 1, math.min(RACING_LINE_CACHE.point_count, 500) do
    local pt = RACING_LINE_CACHE.points[i]
    if pt then
      f:write(string.pack("<fff", pt.x, pt.y, pt.z))
    end
  end
  
  f:close()
  ac.log("[RL_Racing_Line] Wrote " .. RACING_LINE_CACHE.point_count .. " racing line points to cache")
  return true
end

-- Load racing line from track data
local function load_racing_line_for_track()
  local track_name = ac.getTrackName()
  local track_layout = ac.getTrackLayout() or "default"
  
  -- If track hasn't changed, use cached data
  if RACING_LINE_CACHE.last_track == track_name and 
     RACING_LINE_CACHE.last_layout == track_layout and
     RACING_LINE_CACHE.point_count > 0 then
    return true
  end
  
  -- Clear previous racing line
  RACING_LINE_CACHE.points = {}
  RACING_LINE_CACHE.point_count = 0
  
  -- Try to get racing line points from AC
  local success = false
  local points_added = 0
  
  -- Method 1: Use ac.getRacelinePoint() if available
  if ac.getRacelinePoint then
    local max_index = 2000  -- safety limit
    for idx = 0, max_index do
      local point = ac.getRacelinePoint(idx)
      if not point then
        break
      end
      table.insert(RACING_LINE_CACHE.points, point)
      points_added = points_added + 1
      if points_added >= 500 then
        break
      end
    end
    
    if points_added > 0 then
      RACING_LINE_CACHE.point_count = points_added
      RACING_LINE_CACHE.last_track = track_name
      RACING_LINE_CACHE.last_layout = track_layout
      ac.log("[RL_Racing_Line] Loaded " .. points_added .. " points for track: " .. track_name)
      success = true
    end
  end
  
  -- Method 2: Fallback - manually sample track at intervals
  if not success and ac.getTrackLength then
    local track_length = ac.getTrackLength()
    if track_length and track_length > 0 then
      local sample_interval = 2.0  -- Sample every 2 meters
      
      for distance = 0, track_length - sample_interval, sample_interval do
        -- Approximate racing line position (use car position on track as proxy)
        -- This is a fallback - ideally use getRacelinePoint above
        local prog = distance / track_length
        
        -- Simple linear interpolation placeholder
        local point = {
          x = distance,
          y = 0,
          z = 0
        }
        table.insert(RACING_LINE_CACHE.points, point)
        points_added = points_added + 1
        
        if points_added >= 500 then
          break
        end
      end
      
      if points_added > 0 then
        RACING_LINE_CACHE.point_count = points_added
        RACING_LINE_CACHE.last_track = track_name
        RACING_LINE_CACHE.last_layout = track_layout
        ac.log("[RL_Racing_Line] Loaded " .. points_added .. " fallback points for track: " .. track_name)
        success = true
      end
    end
  end
  
  if not success then
    ac.log("[RL_Racing_Line] Could not load racing line for track: " .. track_name)
    return false
  end
  
  -- Write to cache file for Python
  write_racing_line_cache()
  return true
end
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
  
  -- Check toggle file state
  local toggle_state = readFile(TOGGLE_FILE)
  if not toggle_state then
    -- File doesn't exist yet, create it with "0" state
    local created = writeFile(TOGGLE_FILE, "0")
    if created then
      ac.log("[RL_Reset] Created toggle file: " .. TOGGLE_FILE)
    else
      ac.log("[RL_Reset] ERROR: Could not create toggle file")
    end
    return
  end
  
  -- Strip whitespace for robust comparison
  toggle_state = toggle_state:match("^%s*(.-)%s*$") or toggle_state
  
  -- Check if reset is requested (state = "1")
  if toggle_state == "1" or toggle_state:match("^1") then
    ac.log("[RL_Reset] Reset command received, toggle state: '" .. toggle_state .. "'")
    
    -- Perform the reset
    local success = resetCar()
    
    -- CRITICAL: Load racing line immediately after reset
    -- This ensures Python can read it in the next step
    if success then
      ac.log("[RL_Reset] Car reset successful, loading racing line...")
      load_racing_line_for_track()
      
      -- Write "0" back to confirm reset complete
      local written = writeFile(TOGGLE_FILE, "0")
      if written then
        ac.log("[RL_Reset] Reset complete, toggle cleared to '0'")
      else
        ac.log("[RL_Reset] ERROR: Could not write '0' to toggle file - write may have failed!")
      end
    else
      -- Reset failed, still clear toggle to avoid infinite loop
      ac.log("[RL_Reset] Car reset FAILED - clearing toggle anyway")
      writeFile(TOGGLE_FILE, "0")
    end
  end
end

-- Session start - verify script is running
function script.sessionStart()
  ac.log("[RL_Reset] ========================================")
  ac.log("[RL_Reset] CSP Reset Script LOADED and RUNNING")
  ac.log("[RL_Reset] Toggle file: " .. TOGGLE_FILE)
  ac.log("[RL_Reset] Racing line cache: " .. RACING_LINE_CACHE_FILE)
  ac.log("[RL_Reset] ========================================")
  
  -- Load racing line at session start
  ac.log("[RL_Racing_Line] Session started, loading racing line")
  load_racing_line_for_track()
  
  -- Initialize toggle file
  writeFile(TOGGLE_FILE, "0")
  ac.log("[RL_Reset] Toggle file initialized to '0'")
end
