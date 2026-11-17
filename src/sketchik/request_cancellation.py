"""Interrupt one HTTP exchange without abandoning a worker or replaying it."""
import http.client
import socket
from threading import Event, Lock
import time
import urllib.request


class RequestCancelled(InterruptedError):
    """Intentional cancellation, never a retryable provider/network failure."""


class RequestCancellation:
    def __init__(self):
        self.event = Event()
        self.lock = Lock()
        self.sockets = []
        self.requested_at = None
        self.reason = None

    def check(self):
        if self.event.is_set():
            raise RequestCancelled('Obsolete advance planning request cancelled')

    def attach(self, sock):
        with self.lock:
            self.sockets.append(sock)
            cancelled = self.event.is_set()
        if cancelled:
            self._shutdown(sock)
        try:
            self.check()
        except RequestCancelled:
            sock.close()
            raise
        return sock

    @staticmethod
    def _shutdown(sock):
        # Closing a buffered HTTPResponse from another thread can itself block.
        # Shutdown wakes its reader; the owning worker closes all resources.
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def cancel(self, reason):
        with self.lock:
            if self.requested_at is None:
                self.requested_at = time.perf_counter()
                self.reason = reason
            self.event.set()
            sockets = tuple(self.sockets)
        for sock in sockets:
            self._shutdown(sock)

    def create_connection(self, *args, **kwargs):
        self.check()
        # DNS/connect retain the configured transport timeout; once connected,
        # headers, request upload and response reads can all be interrupted.
        return self.attach(socket.create_connection(*args, **kwargs))

    def opener(self, *, direct):
        cancellation = self

        class HTTPConnection(http.client.HTTPConnection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._create_connection = cancellation.create_connection

        class HTTPSConnection(http.client.HTTPSConnection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._create_connection = cancellation.create_connection

            def connect(self):
                http.client.HTTPConnection.connect(self)
                cancellation.check()
                hostname = self._tunnel_host or self.host
                self.sock = cancellation.attach(self._context.wrap_socket(
                    self.sock, server_hostname=hostname, do_handshake_on_connect=False))
                self.sock.do_handshake()
                cancellation.check()

        class HTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, req):
                cancellation.check()
                return self.do_open(HTTPConnection, req)

        class HTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, req):
                cancellation.check()
                return self.do_open(HTTPSConnection, req,
                    context=self._context, check_hostname=self._check_hostname)

        handlers = [HTTPHandler(), HTTPSHandler()]
        if direct:
            handlers.insert(0, urllib.request.ProxyHandler({}))
        return urllib.request.build_opener(*handlers)
