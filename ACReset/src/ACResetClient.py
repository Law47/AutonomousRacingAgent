import socket
import errno

class ACResetClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self._connected = False
        self._rx_buf = bytearray()
        self._create_socket()

    def _create_socket(self):
        """Create (or recreate) the non-blocking TCP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)

    def connect_nonblocking(self):
        """Try to establish a connection without blocking.

        On Windows a failed non-blocking connect leaves the socket in an
        unusable state, so we recreate it before every new attempt.
        """
        if self._connected:
            return True

        try:
            self.sock.connect((self.host, self.port))
            self._connected = True
            return True
        except BlockingIOError:
            # connect in progress — will complete on a later tick
            return False
        except OSError as e:
            if e.errno == errno.EISCONN:
                self._connected = True
                return True
            # Connection refused / reset / other failure:
            # On Windows the socket is now unusable, so recreate it.
            try:
                self.sock.close()
            except Exception:
                pass
            self._create_socket()
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
                        self._create_socket()
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