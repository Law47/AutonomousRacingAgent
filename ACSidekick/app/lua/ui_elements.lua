---@class UIElement
---@field pos vec2
---@field size vec2
---@field padding vec2
---@field color rgbm
---@field onClick boolean
---@field internalElements table[]
UIElement = {}
UIElement.__index = UIElement

function UIElement:new(pos, size)
    local uiElement = {}
    setmetatable(uiElement, UIElement)
    uiElement.pos = pos
    uiElement.size = size
    uiElement.internalElements = {}
    uiElement.color = rgbm(0.1, 0.1, 0.1, 1)
    uiElement.onClick = false
    return uiElement
end

---@param color rgbm
function UIElement:setBackground(color)
    self.color = color
end

function UIElement:addText(text, size, p1, p2)
    local textElement = {
        type = "textElement",
        text = text,
        fontSize = size,
        p1 = p1,
        p2 = p2
    }
    table.insert(self.internalElements, textElement)
end

function UIElement:addImage(imageSrc, p1, p2, rotation, pivot)
    local imageElement = {
        type = "imageElement",
        imageSrc = imageSrc,
        p1 = p1,
        p2 = p2,
        rotation = rotation,
        pivot = pivot
    }
    table.insert(self.internalElements, imageElement)
end

---@param element UIElement
function UIElement:addElement(element)
    local uiElement = {
        type = "uiElement",
        element = element
    }
end

function UIElement:clearElements()
    table.clear(self.internalElements)
end

function UIElement:onMouseEnter()

end

function UIElement:onMouseLeave()
    
end

function UIElement:draw()
    ui.transparentWindow("Element", self.pos, self.size, function()
        ui.drawRectFilled(vec2(0, 0), self.size, self.color)
        for _, element in pairs(self.internalElements) do
            if element.type == "textElement" then
                ui.dwriteDrawTextClipped(element.text, element.fontSize, element.p1, element.p2, ui.Alignment.Center, ui.Alignment.Center)
            elseif element.type == "imageElement" then
                ui.beginRotation()
                ui.drawImage(element.imageSrc, element.p1, element.p2)
                ui.endPivotRotation(element.rotation, element.pivot)
            elseif element.type == "uiElement" then
                element.drawWithOffset(self.pos)
            end
        end
    end)
end

---@param offset vec2
function UIElement:drawWithOffset(offset)
    local oldPos = self.pos
    self.pos = self.pos + offset
    self:draw()
    self.pos = oldPos
end