import socket, time

HOST = "127.0.0.1"
PORT = 65432

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))
def sendReset(srv):
    srv.listen(1)
    conn, addr = srv.accept()
    time.sleep(1)
    conn.sendall(b"RESET\n")

sendReset(srv)
print("reset")