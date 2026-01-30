import socket
import threading

def handle_client(conn, address):
    print(f"Client connected: {address}")
    try:
        with conn:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                try:
                    text = data.decode(errors='replace')
                except Exception:
                    text = str(data)
                    print(f"[{address[0]}:{address[1]}] {text}")
    except Exception as e:
        print(f"Connection handler error for {address}: {e}")
    finally:
        print(f"Client disconnected: {address}")


def server_program(host='0.0.0.0', port=8000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"Server listening on {host}:{port}")
    try:
        while True:
            conn, address = server_socket.accept()
            t = threading.Thread(target=handle_client, args=(conn, address), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("Server shutting down")
    finally:
        server_socket.close()


if __name__ == '__main__':
    server_program()
