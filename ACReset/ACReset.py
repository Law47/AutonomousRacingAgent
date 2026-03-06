#Required For AC APPS
import ac
import acsys
from third_party.sim_info import *

from src.Utils import *
from src.ACResetClient import *

#AC Docs https://docs.google.com/document/d/13trBp6K1TjWbToUQs_nfFsB291-zVJzRZCNaTYt4Dzc/pub

appName = "ACReset"
width, height = 200 , 200

window = None
dynamicElements = []
clickableElements = []
simInfo = SimInfo()

red = Color(1, 0, 0, 1)

centerPos = Vector2(width/2, height/2)

HOST = "127.0.0.1"
PORT = 65432

clientInitalized = False
resetClient = None

def acMain(ac_version):
    global appWindow # <- you'll need to update your window in other functions.

    appWindow = ac.newApp(appName)
    ac.setSize(appWindow, width, height)
    ac.drawBorder(appWindow, 0)
    ac.setBackgroundOpacity(appWindow, 0)
    ac.setTitlePosition(appWindow, 10000, 10000)
    ac.setIconPosition(appWindow, 10000, 10000)

    ac.addRenderCallback(appWindow, appGL) # -> links this app's window to an OpenGL render function

    global window
    window = Element(appWindow, centerPos, Vector2(width, height), "", 24, Color(0, 0, 0, 0.5))
    window.expandEffect = False
    ac.addOnClickedListener(window.ref, windowOnClick)

    ResetButton = Element(appWindow, centerPos, Vector2(100, 50), "reset", 24, red, id="Reset")
    dynamicElements.append(ResetButton)
    clickableElements.append(ResetButton)

    topBar = Element(appWindow, Vector2(width/2, 11), Vector2(width, 22), "ACReset", 16, Color(0.25, 0.25, 0.25, 1))

    global resetClient
    global clientInitalized
    resetClient = ACResetClient(HOST, PORT)
    clientInitalized = True
    ac.log("Socket Initalized")
    
    return appName

def appGL(deltaT):#-------------------------------- OpenGL UPDATE
    """
    This is where you redraw your openGL graphics
    if you need to use them .
    """
    pass # -> Delete this line if you do something here !

def windowOnClick(x, y):
    clickPos = Vector2(x, y)
    for element in clickableElements:
        if element.position - (element.size/2) < clickPos and element.position + (element.size/2) > clickPos:
            element.onClick()

            if element.id == "Reset":
                reset()

def acUpdate(deltaT):#-------------------------------- AC UPDATE
    global currentTime
    global nextInsertTimestamp

    ac.setBackgroundOpacity(appWindow, 0)

    for element in dynamicElements:
        element.update(deltaT)
    
    if clientInitalized:
        msg = resetClient.requestMessage()

        if msg == "Empty":
            return
        
        if msg == "NoConn":
            return

        if msg == "RESET":
            reset()
            return
        
        ac.log("Got Message: " + str(msg))

def reset():
    ac.log("Resetting ...")
    ac.ext_resetCar()