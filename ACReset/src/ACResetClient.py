# ACReset.py
import socket
import errno

class ACResetClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)

        self._connected = False
        self._rx_buf = bytearray()

    def connect_nonblocking(self):
        """Call once at init; then keep calling from update until connected."""
        if self._connected:
            return True

        try:
            # Non-blocking connect: usually raises immediately with EINPROGRESS
            self.sock.connect((self.host, self.port))
            self._connected = True
            return True
        except BlockingIOError:
            # connect in progress
            return False
        except OSError as e:
            # Already connected can show up as EISCONN on some platforms
            if e.errno == errno.EISCONN:
                self._connected = True
                return True
            # Connection refused / not ready: swallow and retry later
            return False

    def requestMessage(self):
        # 1) ensure connection without blocking
        if not self._connected and not self.connect_nonblocking():
            return "NoConn"

        # 2) read whatever is available without blocking
        while True:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    # server closed
                    self._connected = False
                    try:
                        self.sock.close()
                    finally:
                        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.sock.setblocking(False)
                    return "Empty"

                self._rx_buf.extend(chunk)

            except BlockingIOError:
                # no more data available this tick
                break
            except OSError:
                # treat as disconnect
                self._connected = False
                break

        # 3) parse messages (example: newline-delimited commands)
        return self._process_messages()

    def _process_messages(self):
        nl = self._rx_buf.find(b"\n")
        if nl == -1:
            return "Empty"

        line = bytes(self._rx_buf[:nl]).strip()
        del self._rx_buf[:nl + 1]

        try:
            return line.decode("utf-8", errors="replace")
        except Exception:
            return "Empty"