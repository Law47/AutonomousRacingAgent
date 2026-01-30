---@diagnostic disable: duplicate-set-field

UIElements = require(".app/lua/ui_elements")
TelemetryCollector = require(".app/lua/telemetry_collector")
Util = require(".app/lua/util")
-- Load LuaSocket (Assetto Corsa's Lua may return `true` instead of a table).
local _sockmod = require("socket")
if type(_sockmod) == "table" then
  Socket = _sockmod
else
  Socket = {}
  if type(connect) == "function" then Socket.connect = connect end
  if type(bind) == "function" then Socket.bind = bind end
  if type(source) == "function" then Socket.source = source end
  if type(sink) == "function" then Socket.sink = sink end
  if type(socket) == "table" then
    Socket.core = socket
    if type(socket.tcp) == "function" then Socket.tcp = socket.tcp end
  end
end

initalized = false
guiInitalized = false
collectingData = false
windowActive = false

time = 0
uploadPacketInterval = 1/24
uploadPacketStep = uploadPacketInterval

csvFilePath = "apps/lua/ACSidekick/TelemetryOutputs/"
csvFileNameDefault = "TelemetryOutput.csv"
csvFileName = nil
writtenToCSV = false
csvFile = nil

---@type UIElement[]
visibleElements = {}

---@type UIElement[]
clickableElements = {}

function script.onShowWindow()
  -- Window Opened
  windowActive = true

  if not initalized then
    -- Load configs/defaultValues
    ini = ac.getFolder(ac.FolderID.ContentTracks) .. '/' .. ac.getTrackFullID('/') .. '/data/map.ini'
    config = ac.INIConfig.load(ini):mapSection('PARAMETERS', { SCALE_FACTOR = 1, Z_OFFSET = 1, X_OFFSET = 1, WIDTH=500, HEIGHT=500, MARGIN=20, DRAWING_SIZE=10, MAX_SIZE=1000})
    config.OFFSETS = vec2(config.X_OFFSET, config.Z_OFFSET)

    uiState = ac.getUI()
    displaySize = uiState.windowSize
    playerCar = ac.getCar(0)
    
    -- TelemetryCollection
    -- dbPort = 8091

    -- ---@type DBLink
    -- dbLink = DBLink:new(dbPort)

    addGUIElements()
    initalized = true
  end
end

function script.onHideWindow()
  -- Window Close
  windowActive = false
end

function script.onWindowUpdate(dt)
  -- On Window Update

  if not initalized then return end

  getActiveWindow()

  windowSize = AppInfo.size
  position = AppInfo.position
  defaultTitleFontSize = (windowSize.x * 0.1 + windowSize.y * 0.9)/32
  
  -- Handle onClicks
  if collectDataButton.onClick then
    collectingData = not collectingData

    if collectingData and not writtenToCSV then
      playerCar = ac.getCar(0)
      if not playerCar then return end
      packet = CreatePacket(0, 0, 0, "'" .. os.date("%m/%d/%Y %X", os.time()) .. string.format(":%d", (time - math.floor(time))*1000) .. "'", playerCar)

      -- createCSVFile(packet)
    end
  end

  -- Update GUI
  dataCollectionDisplay.pos = position
  dataCollectionDisplay.size = vec2(windowSize.x, windowSize.y * 0.5)
  dataCollectionDisplay:clearElements()

  dataCollectionDisplay:addText("Timing", defaultTitleFontSize, vec2(0, 0), vec2(windowSize.x, 50))
  dataCollectionDisplay:addText("Packets Written: " .. packetsSent, defaultTitleFontSize/1.5, vec2(0, dataCollectionDisplay.size.y * 0.9), dataCollectionDisplay.size)

  collectDataButton.size = vec2(dataCollectionDisplay.size.x/3, dataCollectionDisplay.size.y/8)
  collectDataButton.pos = position + dataCollectionDisplay.size/2 - collectDataButton.size/2 + vec2(0, windowSize.y*0.175)
  collectDataButton:clearElements()

  if collectingData then 
    collectDataButton:addText("Unlink DB", defaultTitleFontSize/1.5, vec2(0, 0), collectDataButton.size)
    collectDataButton:setBackground(rgbm(0.3, 0, 0, 1))
  else
    collectDataButton:addText("Link DB", defaultTitleFontSize/1.5, vec2(0, 0), collectDataButton.size)
    collectDataButton:setBackground(rgbm(0, 0.3, 0, 1))
  end

  mapDisplay.pos = position + vec2(0, dataCollectionDisplay.size.y)
  mapDisplay.size = vec2(windowSize.x, windowSize.y - dataCollectionDisplay.size.y)
  mapDisplay:clearElements()

  -- Map positioning
  local mapSizeFactor = math.pow(1.15, mapScale)
  local mapOffset = (vec2(playerCar.position.x, playerCar.position.z) + config.OFFSETS) * (mapSizeFactor/config.SCALE_FACTOR) - mapDisplay.size/2
  rotationangle = 270 - math.deg(math.atan2(playerCar.look.x, playerCar.look.z))
  local p1 = -mapOffset
  local p2 = -mapOffset + (mapSizeFactor * mapImageSize)
  mapDisplay:addImage(mapImageSrc, p1, p2, rotationangle, mapDisplay.size/2)
  mapDisplay:addImage("./data/Arrow.png", mapDisplay.size/2 - vec2(25, 25), mapDisplay.size/2 + vec2(25, 25), 90, mapDisplay.size/2)

  guiInitalized = true

  time = time + dt
end

function script.update(dt)
  -- On Script Update 

  if not initalized then return end

  if collectingData then
    if time > uploadPacketStep then
      playerCar = ac.getCar(0)
      if not playerCar then return end

      packet = CreatePacket(0, 0, 0, "'" .. os.date("%m/%d/%Y %X", os.time()) .. string.format(":%d", (time - math.floor(time))*1000) .. "'", playerCar)

      uploadPacketStep = time + uploadPacketInterval

      -- if not csvFile then
      --   createCSVFile(packet)
      -- end

      -- addToCSVFile(packet)

      sendPacketToSocket(packet)

      packetsSent = packetsSent + 1
    end
  end

  -- Write up your own backend with golang to store the telemetry data with SQLite
  
end

function script.scenePreRenderUpdate()
  -- Called before a scene has started rendering
end

function script.postGeometryRenderUpdate()
  -- Called when opaque geometry has finished rendering
end

function script.preRenderUIUpdate()
  -- Called before rendering ImGui apps to draw things on screen

  if windowActive then
    if not AppInfo then
      getActiveWindow()
    end

    uiState = ac.getUI()
    position = AppInfo.position;

    if guiInitalized then
      -- map scaling update
      if uiState.mouseWheel ~= 0 then
        if (uiState.mousePos.x > position.x and uiState.mousePos.y > position.y and uiState.mousePos.x < (position + AppInfo.size).x and uiState.mousePos.y < (position + AppInfo.size).y) then
          mapScale = mapScale + uiState.mouseWheel
        end
      end
      
      ui.pushDWriteFont(ui.DWriteFont("Chakra Petch;Weight=Light;", "./data"))

      for _, element in pairs(visibleElements) do
        element:draw()
      end

      for _, element in pairs(clickableElements) do
        if (uiState.isMouseLeftKeyClicked) then
          if (uiState.mousePos.x > element.pos.x and uiState.mousePos.y > element.pos.y and uiState.mousePos.x < (element.pos + element.size).x and uiState.mousePos.y < (element.pos + element.size).y) then
            element.onClick = true
          else
            element.onClick = false
          end
        else
          element.onClick = false
        end
      end
    end
  end
end

function script.simulationUpdate()
  -- Called after a whole simulation update
end

function addGUIElements()
    ---@type UIElement
  dataCollectionDisplay = UIElement:new()
  dataCollectionDisplay:setBackground(rgbm(0.1, 0.1, 0.1, 1))
  table.insert(visibleElements, dataCollectionDisplay)

  ---@type UIElement
  collectDataButton = UIElement:new()
  table.insert(visibleElements, collectDataButton)
  table.insert(clickableElements, collectDataButton)
  packetsSent = 0

  ---@type UIElement
  mapDisplay = UIElement:new()
  mapDisplay:setBackground(rgbm(0.1, 0.1, 0.1, 1))
  table.insert(visibleElements, mapDisplay)

  local map_mini = ac.getFolder(ac.FolderID.ContentTracks) .. '\\' .. ac.getTrackFullID('\\') .. '\\map_mini.png'
  local map = ac.getFolder(ac.FolderID.ContentTracks) .. '\\' .. ac.getTrackFullID('\\') .. '\\map.png'
  mapImageSrc = io.exists(map_mini) and map_mini or map
  mapImageOffset = vec2(0, 50)
  mapImagePadding = vec2(20, 20)
  mapImageSize = ui.imageSize(mapImageSrc)
  mapScale = 1
end

function createCSVFile(packet)
  csvFileName = csvFilePath .. os.date("%S_%M_%H_%d_%m_%Y", os.time()) .. csvFileNameDefault

  csvWriteData = GetCSVHeaders(packet)
  csvFile = io.open(csvFileName, "w")
  csvFile:write(csvWriteData)
  csvFile:close()

  ac.log("Created File " .. csvFileName)
  writtenToCSV = true
end

function addToCSVFile(packet)
  csvWriteData = GetCSVBody(packet)
  csvFile = io.open(csvFileName, "a")
  csvFile:write(csvWriteData)
  csvFile:close()
end

socketSetup = false
tcp_socket = nil

function setupSocketConnection()
  if socketSetup then return end

  local ip = "127.0.0.1"
  local port = 8000

  local _tcp_socket, error_msg = Socket.connect(ip, port)

  if not _tcp_socket then
      ac.log("Failed to connect to Python server: " .. error_msg)
      return
  else
      ac.log("Connected to Python server")
      socketSetup = true
      tcp_socket = _tcp_socket
  end

end

function sendPacketToSocket(packet)
  if not socketSetup then setupSocketConnection() return end

  local packetHeaders = GetCSVHeaders(packet)
  local packetSendData = GetCSVBody(packet)
  local success, send_err = tcp_socket:send(packetHeaders .. "\n" .. packetSendData .. "\n")

  if not success then
    ac.log("Failed to send data: " .. send_err)
    socketSetup = false
  else
    ac.log("Data sent: " .. packetSendData)
  end
end

function getActiveWindow()
  local windows = ac.getAppWindows()
  for _, window in pairs(windows) do
    if window.title == "ACSidekick" then
      AppInfo = window
    end
  end
end

function shutdown(dt)
  ac.load("app is shutting down")

  if tcp_socket then tcp_socket:close() end
end