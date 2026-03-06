import ac;

class Color:
    def __init__(self, r, g, b, a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, a):
        if type(a) is Vector2:
            return Vector2(self.x + a.x, self.y + a.y)
        elif type(a) is int:
            return Vector2(self.x + a, self.y + a)
        elif type(a) is float:
            return Vector2(self.x + a, self.y + a)
        return None

    def __sub__(self, a):
        if type(a) is Vector2:
            return Vector2(self.x - a.x, self.y - a.y)
        elif type(a) is int:
            return Vector2(self.x - a, self.y - a)
        elif type(a) is float:
            return Vector2(self.x - a, self.y - a)
        return None
    
    def __truediv__(self, a):
        if type(a) is int:
            return Vector2(self.x/a, self.y/a)
        elif type(a) is float:
            return Vector2(self.x/a, self.y/a)
        return None
        
    def __mul__(self, a):
        if type(a) is int:
            return Vector2(self.x*a, self.y*a)
        elif type(a) is float:
            return Vector2(self.x*a, self.y*a)
        return None
    
    def __gt__(self, a):
        if type(a) is Vector2:
            return self.x > a.x and self.y > a.y
        return None
    
    def __lt__(self, a):
        if type(a) is Vector2:
            return self.x < a.x and self.y < a.y
        return None

class DynamicUI:
    def __init__(self):
        self.time = 0

    def update(self, deltaT):
        self.time += deltaT

class UIElements:
    def __init__(self, appWindow, ref, position, size = None):
        self.appWindow = appWindow
        self.position = position
        self.size = size
        self.ref = ref
        self.autoCentering = True
    
    def updatePosition(self):
        if (self.size != None and self.autoCentering):
            ac.setPosition(self.ref, self.position.x - self.size.x/2, self.position.y - self.size.y/2)
            ac.setSize(self.ref, self.size.x, self.size.y)
        else:
            ac.setPosition(self.ref, self.position.x, self.position.y)

class Image(UIElements):
    def __init__(self, appWindow, position, size, imageSource):
        UIElements.__init__(self, appWindow, ac.addGraph(appWindow, ""), position, size)
        self.updatePosition()

        ac.setBackgroundTexture(self.ref, imageSource)
        ac.setBackgroundOpacity(self.ref, 0)
        ac.drawBorder(self.ref, 0)

class Text(UIElements):
    def __init__(self, appWindow, position, text, fontSize):
        UIElements.__init__(self, appWindow, ac.addLabel(appWindow, text), position)
        self.updatePosition()

        ac.setFontSize(self.ref, fontSize)
        ac.setFontAlignment(self.ref, "center")

class Element(UIElements, DynamicUI):
    def __init__(self, appWindow, position, size, text, fontSize, texture = None, id = None):
        UIElements.__init__(self, appWindow, ac.addButton(appWindow, text), position, size)
        DynamicUI.__init__(self)
        self.defaultSize = size
        self.updatePosition()

        self.id = id

        ac.setFontSize(self.ref, fontSize)
        ac.setFontAlignment(self.ref, "center")
        ac.setBackgroundOpacity(self.ref, 0)
        ac.drawBorder(self.ref, 0)

        if (texture != None):
            if (type(texture) is str):
                self.texture = texture
                ac.setBackgroundTexture(self.ref, texture)
            if (type(texture) is Color):
                self.color = texture
                ac.setBackgroundColor(self.ref, texture.r, texture.g, texture.b)
                ac.setBackgroundOpacity(self.ref, texture.a)

        self.expandEffect = True
        self.buttonPressed = False
        self.buttonClickEffectSizeChange = 5
        self.buttonClickEffectDuration = 0.1
        self.buttonClickEffectStep = self.time + self.buttonClickEffectDuration

    def updatePosition(self):
        ac.setPosition(self.ref, self.position.x - self.size.x/2, self.position.y - self.size.y/2)
        ac.setSize(self.ref, self.size.x, self.size.y)

    def onClick(self):
        if self.expandEffect:
            self.size = Vector2(self.size.x + self.buttonClickEffectSizeChange, self.size.y + self.buttonClickEffectSizeChange)
            self.updatePosition()
            self.buttonPressed = True
            self.buttonClickEffectStep = self.time + self.buttonClickEffectDuration

    def update(self, deltaT):
        self.time += deltaT

        if (self.expandEffect and self.buttonPressed and self.time > self.buttonClickEffectStep):
            self.buttonPressed = False
            self.size = self.defaultSize
            self.updatePosition()

class Utils:
    @staticmethod
    def log(x):
        ac.log(str(x))