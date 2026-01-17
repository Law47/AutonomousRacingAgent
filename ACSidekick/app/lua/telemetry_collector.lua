---@class DBLink
---@field port number
DBLink = {}
DBLink.__index = DBLink

---@param port number
function DBLink:new(port)
    local dbLink = {}
    setmetatable(dbLink, DBLink)
    dbLink.port = port
    return dbLink
end

---@param packet table
function DBLink:uploadPacket(packet)
    url = string.format("http://localhost:%d/appendTelemetryCSV", self.port)
    body = ToCSV(packet)

    ac.log(body)

    header = {
        ["Content-Type"] = "text/csv"
    }
    packetID = packet.PacketID
    outputString = string.format("\nInserting into DB at time: %0.3f", time)

    web.post(url, header, body, function (err, response)
        outputString = outputString .. string.format("\n\nResponse:\n%s", response.body)
        if err ~= nil then
            outputString = outputString .. string.format("\n\nError:\n%s", err.dump())
        end
        ac.log(outputString)
    end)
    

end

---@param packet table
function GetCSVHeaders(packet)
    csvHeaders = ""
    for k, _ in pairs(packet) do
        csvHeaders = csvHeaders .. string.format("%s,", tostring(k))
    end
    return string.sub(csvHeaders, 1, -2) .. "\n"
end

---@param packet table
function GetCSVBody(packet)
    csvData = ""
    for _, v in pairs(packet) do
        if (type(v) == "number") then
            csvData = csvData .. string.format("%s,", string.format("%s", v))
        else
            csvData = csvData .. string.format("%s,", v)
        end
    end
    return string.sub(csvData, 1, -2) .. "\n"
end

---@param packet table
function ToCSV(packet)
    csvHeaders = ""
    csvData = ""
    for k, v in pairs(packet) do
        csvHeaders = csvHeaders .. string.format("%s,", tostring(k))
        if (type(v) == "number") then
            csvData = csvData .. string.format("%s,", string.format("%s", v))
        else
            csvData = csvData .. string.format("%s,", v)
        end
    end
    return string.sub(csvHeaders, 1, -2) .. "\n" .. string.sub(csvData, 1, -2)
end

---@param packetID number
---@param sessionID number
---@param lapID number
---@param time string
---@param data ac.StateCar
function CreatePacket(packetID, sessionID, lapID, time, data)
    local packet = {}
    
    -- PacketInfo
    packet.PacketID = packetID
    packet.SessionID = sessionID
    packet.LapID = lapID
    packet.PacketDatetime = time

    -- DriverInputs
    packet.Gas = data.gas
    packet.Brake = data.brake
    packet.Steer = data.steer
    packet.Clutch = data.clutch
    packet.Handbrake = data.handbrake
    packet.Gear = data.gear

    -- CarInfo
    packet.Fuel = data.fuel
    packet.SpeedMPH = data.speedMs
    packet.RPM = data.rpm
    packet.EngagedGear = data.engagedGear
    packet.TurboBoost = data.turboBoost
    packet.Weight = data.mass
    packet.WorldPositionX = data.position.x
    packet.WorldPositionY = data.position.y
    packet.WorldPositionZ = data.position.z
    packet.AngularVelocityX = data.angularVelocity.x
    packet.AngularVelocityY = data.angularVelocity.y
    packet.AngularVelocityZ = data.angularVelocity.z
    packet.VelocityX = data.velocity.x
    packet.VelocityY = data.velocity.y
    packet.VelocityZ = data.velocity.z
    packet.AccelerationX = data.acceleration.x
    packet.AccelerationY = data.acceleration.y
    packet.AccelerationZ = data.acceleration.z
    packet.Aero_DragCoeffcient = data.aeroDrag
    packet.Aero_LiftCoefficientFront = data.aeroLiftFront
    packet.Aero_LiftCoefficientRear = data.aeroLiftRear
    packet.CarForwardVectorX = data.look.x
    packet.CarForwardVectorY = data.look.y
    packet.CarForwardVectorZ = data.look.z
    packet.CarSideVectorX = data.side.x
    packet.CarSideVectorY = data.side.y
    packet.CarSideVectorZ = data.side.z

    -- ACState
    packet.ResetCount = data.resetCounter
    packet.CollidedWith = data.collidedWith
    packet.HeadlightsActive = data.headlightsActive
    packet.Ping = data.ping
    packet.SteerTorque = data.steerTorque

    wheelData = playerCar.wheels

    -- TyreInfo
    packet.FL_Camber = wheelData[0].camber
    packet.FR_Camber = wheelData[1].camber
    packet.RL_Camber = wheelData[2].camber
    packet.RR_Camber = wheelData[3].camber
    packet.FL_ToeIn = wheelData[0].toeIn
    packet.FR_ToeIn = wheelData[1].toeIn
    packet.RL_ToeIn = wheelData[2].toeIn
    packet.RR_ToeIn = wheelData[3].toeIn
    packet.FL_TyreRadius = wheelData[0].tyreRadius
    packet.FR_TyreRadius = wheelData[1].tyreRadius
    packet.RL_TyreRadius = wheelData[2].tyreRadius
    packet.RR_TyreRadius = wheelData[3].tyreRadius
    packet.FL_TyreWidth = wheelData[0].tyreWidth
    packet.FR_TyreWidth = wheelData[1].tyreWidth
    packet.RL_TyreWidth = wheelData[2].tyreWidth
    packet.RR_TyreWidth = wheelData[3].tyreWidth
    packet.FL_RimRadius = wheelData[0].rimRadius
    packet.FR_RimRadius = wheelData[1].rimRadius
    packet.RL_RimRadius = wheelData[2].rimRadius
    packet.RR_RimRadius = wheelData[3].rimRadius

    -- TyreState
    packet.FL_TyreWear = wheelData[0].tyreWear
    packet.FR_TyreWear = wheelData[1].tyreWear
    packet.RL_TyreWear = wheelData[2].tyreWear
    packet.RR_TyreWear = wheelData[3].tyreWear
    packet.FL_TyreVirtualMPH = wheelData[0].tyreVirtualKM * 0.62
    packet.FR_TyreVirtualMPH = wheelData[1].tyreVirtualKM * 0.62
    packet.RL_TyreVirtualMPH = wheelData[2].tyreVirtualKM * 0.62
    packet.RR_TyreVirtualMPH = wheelData[3].tyreVirtualKM * 0.62
    packet.FL_TyreDirtyLevel = wheelData[0].tyreDirty
    packet.FR_TyreDirtyLevel = wheelData[1].tyreDirty
    packet.RL_TyreDirtyLevel = wheelData[2].tyreDirty
    packet.RR_TyreDirtyLevel = wheelData[3].tyreDirty
    packet.FL_Slip = wheelData[0].slip
    packet.FR_Slip = wheelData[1].slip
    packet.RL_Slip = wheelData[2].slip
    packet.RR_Slip = wheelData[3].slip
    packet.FL_SlipAngle = wheelData[0].slipAngle
    packet.FR_SlipAngle = wheelData[1].slipAngle
    packet.RL_SlipAngle = wheelData[2].slipAngle
    packet.RR_SlipAngle = wheelData[3].slipAngle
    packet.FL_SlipRatio = wheelData[0].slipRatio
    packet.FR_SlipRatio = wheelData[1].slipRatio
    packet.RL_SlipRatio = wheelData[2].slipRatio
    packet.RR_SlipRatio = wheelData[3].slipRatio
    packet.FL_NDSlip = wheelData[0].ndSlip
    packet.FR_NDSlip = wheelData[1].ndSlip
    packet.RL_NDSlip = wheelData[2].ndSlip
    packet.RR_NDSlip = wheelData[3].ndSlip
    packet.FL_Load = wheelData[0].loadK
    packet.FR_Load = wheelData[1].loadK
    packet.RL_Load = wheelData[2].loadK
    packet.RR_Load = wheelData[3].loadK
    packet.FL_CoreTemperature = wheelData[0].tyreCoreTemperature
    packet.FR_CoreTemperature = wheelData[1].tyreCoreTemperature
    packet.RL_CoreTemperature = wheelData[2].tyreCoreTemperature
    packet.RR_CoreTemperature = wheelData[3].tyreCoreTemperature
    packet.FL_TyreInsideTemperature = wheelData[0].tyreInsideTemperature
    packet.FR_TyreInsideTemperature = wheelData[1].tyreInsideTemperature
    packet.RL_TyreInsideTemperature = wheelData[2].tyreInsideTemperature
    packet.RR_TyreInsideTemperature = wheelData[3].tyreInsideTemperature
    packet.FL_TyreMiddleTemperature = wheelData[0].tyreMiddleTemperature
    packet.FR_TyreMiddleTemperature = wheelData[1].tyreMiddleTemperature
    packet.RL_TyreMiddleTemperature = wheelData[2].tyreMiddleTemperature
    packet.RR_TyreMiddleTemperature = wheelData[3].tyreMiddleTemperature
    packet.FL_TyreOutsideTemperature = wheelData[0].tyreOutsideTemperature
    packet.FR_TyreOutsideTemperature = wheelData[1].tyreOutsideTemperature
    packet.RL_TyreOutsideTemperature = wheelData[2].tyreOutsideTemperature
    packet.RR_TyreOutsideTemperature = wheelData[3].tyreOutsideTemperature
    packet.FL_TyreOptimumTemperature = wheelData[0].tyreOptimumTemperature
    packet.FR_TyreOptimumTemperature = wheelData[1].tyreOptimumTemperature
    packet.RL_TyreOptimumTemperature = wheelData[2].tyreOptimumTemperature
    packet.RR_TyreOptimumTemperature = wheelData[3].tyreOptimumTemperature
    packet.FL_TemperatureMultiplier = wheelData[0].temperatureMultiplier
    packet.FR_TemperatureMultiplier = wheelData[1].temperatureMultiplier
    packet.RL_TemperatureMultiplier = wheelData[2].temperatureMultiplier
    packet.RR_TemperatureMultiplier = wheelData[3].temperatureMultiplier
    packet.FL_StaticPressure = wheelData[0].tyreStaticPressure
    packet.FR_StaticPressure = wheelData[1].tyreStaticPressure
    packet.RL_StaticPressure = wheelData[2].tyreStaticPressure
    packet.RR_StaticPressure = wheelData[3].tyreStaticPressure
    packet.FL_DynamicPressure = wheelData[0].tyrePressure
    packet.FR_DynamicPressure = wheelData[1].tyrePressure
    packet.RL_DynamicPressure = wheelData[2].tyrePressure
    packet.RR_DynamicPressure = wheelData[3].tyrePressure
    packet.FL_SelfAligningTorque = wheelData[0].mz
    packet.FR_SelfAligningTorque = wheelData[1].mz
    packet.RL_SelfAligningTorque = wheelData[2].mz
    packet.RR_SelfAligningTorque = wheelData[3].mz
    packet.FL_TyreContactNormalX = wheelData[0].contactNormal.x
    packet.FL_TyreContactNormalY = wheelData[0].contactNormal.y
    packet.FL_TyreContactNormalZ = wheelData[0].contactNormal.z
    packet.FR_TyreContactNormalX = wheelData[1].contactNormal.x
    packet.FR_TyreContactNormalY = wheelData[1].contactNormal.y
    packet.FR_TyreContactNormalZ = wheelData[1].contactNormal.z
    packet.RL_TyreContactNormalX = wheelData[2].contactNormal.x
    packet.RL_TyreContactNormalY = wheelData[2].contactNormal.y
    packet.RL_TyreContactNormalZ = wheelData[2].contactNormal.z
    packet.RR_TyreContactNormalX = wheelData[3].contactNormal.x
    packet.RR_TyreContactNormalY = wheelData[3].contactNormal.y
    packet.RR_TyreContactNormalZ = wheelData[3].contactNormal.z

    return packet
end
