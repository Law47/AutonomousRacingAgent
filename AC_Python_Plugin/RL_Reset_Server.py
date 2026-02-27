"""
RL_Reset_Server - Assetto Corsa Python App
Purpose: Listen on port 2347 for reset commands from training script
When "reset" is received, call ac.ext_resetCar() to reset car to sector start
"""

import socket
import threading
import sys
import os
import ac
import acsys
import logging

# Simple logging
LOG_FILE = "RL_Reset_Server.log"
def log_msg(msg):
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + "\n")
    except:
        pass

log_msg("=== RL_Reset_Server Starting ===")

RESET_PORT = 2347
SERVER_THREAD = None
SERVER_RUNNING = False

def reset_car():
    """Reset the car to sector start via AC's native API"""
    try:
        ac.ext_resetCar()
        log_msg("[RESET] Car reset successfully")
        return True
    except Exception as e:
        log_msg("[RESET ERROR] {}".format(str(e)))
        return False

def manage_socket_thread():
    """Server thread that listens for reset commands"""
    global SERVER_RUNNING
    
    try:
        host = 'localhost'
        port = RESET_PORT
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)  # Non-blocking timeout
        
        log_msg("[SERVER] Listening on {}:{} for reset commands".format(host, port))
        SERVER_RUNNING = True
        
        while SERVER_RUNNING:
            try:
                client_socket, addr = server_socket.accept()
                log_msg("[SERVER] Client connected: {}".format(str(addr)))
                
                try:
                    # Receive command
                    data = client_socket.recv(1024).decode('utf-8').strip()
                    log_msg("[SERVER] Received: '{}'".format(data))
                    
                    if data == "reset":
                        reset_car()
                    else:
                        log_msg("[SERVER] Unknown command: '{}'".format(data))
                    
                    client_socket.close()
                except Exception as e:
                    log_msg("[SERVER] Error handling client: {}".format(str(e)))
                    try:
                        client_socket.close()
                    except:
                        pass
                        
            except socket.timeout:
                # Timeout is normal, allows checking SERVER_RUNNING flag
                pass
            except Exception as e:
                log_msg("[SERVER] Accept error: {}".format(str(e)))
                break
        
        server_socket.close()
        log_msg("[SERVER] Server closed")
        
    except Exception as e:
        log_msg("[SERVER FATAL] {}".format(str(e)))
        SERVER_RUNNING = False

# Start server thread
def acMain(ac_version):
    """Called when AC loads the app"""
    log_msg("acMain() called")
    global SERVER_THREAD
    
    try:
        # Start socket server in background thread
        SERVER_THREAD = threading.Thread(target=manage_socket_thread)
        SERVER_THREAD.daemon = True
        SERVER_THREAD.start()
        log_msg("Server thread started")
    except Exception as e:
        log_msg("[FATAL] Could not start server: {}".format(str(e)))
    
    return "RL_Reset_Server"

def acShutdown():
    """Called when AC exits"""
    global SERVER_RUNNING
    log_msg("Shutdown requested")
    SERVER_RUNNING = False
    log_msg("=== RL_Reset_Server Stopped ===")

def acUpdate(deltaT):
    """Called every frame (optional)"""
    pass
