"""Minimal static HTTP server with Range support (remotezip needs it)."""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        f = open(path, "rb")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            a, b = m.group(1), m.group(2)
            if a == "":
                start, end = size - int(b), size - 1
            else:
                start, end = int(a), int(b) if b else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            f.seek(start)
            self._remaining = end - start + 1
            return f
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = size
        return f

    def copyfile(self, source, outputfile):
        outputfile.write(source.read(self._remaining))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    os.chdir(sys.argv[1])
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), RangeHandler).serve_forever()
