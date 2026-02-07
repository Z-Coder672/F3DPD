#!/usr/bin/env python3
"""
Eufy Security Camera RTSP Streamer with HTTP Server
Streams video from Eufy security camera using RTSP -> HTTP pipeline
"""

import cv2
import os
import sys
import time
import logging
import threading
from typing import Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import urllib.parse
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import shlex
import numpy as np
import re
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eufy_streamer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def _get_local_ip() -> Optional[str]:
    """
    Get the local IP address of the machine.
    Works on networks with and without internet.
    
    Returns:
        Local IP address or None if unable to determine
    """
    try:
        # First try: connect to public DNS (works if internet available)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    
    try:
        # Second try: use a private IP range that doesn't require internet
        # UDP doesn't actually establish a connection, so this works on air-gapped networks
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    
    return None


def _redact_rtsp(rtsp_url: str) -> str:
    """Redact credentials in an rtsp url for logging."""
    try:
        parsed = urllib.parse.urlsplit(rtsp_url)
        if parsed.username or parsed.password:
            netloc = parsed.hostname or ''
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            redacted = parsed._replace(netloc=netloc)
            return urllib.parse.urlunsplit(redacted)
    except Exception:
        pass
    # Fallback: naive masking up to '@'
    return re.sub(r"rtsp://[^@]*@", "rtsp://***:***@", rtsp_url)


def _encode_userinfo(value: str) -> str:
    """URL-encode username/password components for RTSP URLs."""
    return urllib.parse.quote(value, safe="")


def _replace_rtsp_path(url: str, new_path: str) -> str:
    """Return url with path replaced by new_path (must start with '/')."""
    try:
        parsed = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit(parsed._replace(path=new_path))
    except Exception:
        return url


class FFmpegReader:
    """Read frames from RTSP using ffmpeg, outputting MJPEG images over stdout.

    This approach is resilient with explicit TCP transport and timeouts and avoids
    OpenCV's internal buffering quirks on some RTSP cameras.
    """

    def __init__(self, rtsp_url: str, fps: int = 15, quality: int = 5, out_width: int = 640, out_height: int = 480, mode: str = 'raw', rotation_angle: int = 0):
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.quality = max(2, min(31, quality))
        self.out_width = out_width
        self.out_height = out_height
        self.mode = mode  # 'raw' or 'mjpeg'
        self.rotation_angle = rotation_angle # 0, 90, 180, 270 degrees counter-clockwise
        self.proc: Optional[subprocess.Popen] = None
        self.buffer = bytearray()
        self.running = False
        self._stderr_thread = None
        self._restart_count = 0
        self._max_restarts = 10
        self._last_successful_read = time.time()
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

    def start(self) -> bool:
        if self.proc and self.proc.poll() is None:
            return True

        # Check if we've exceeded max restarts
        if self._restart_count >= self._max_restarts:
            logger.error(f"FFmpegReader: Exceeded maximum restart attempts ({self._max_restarts})")
            return False

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-rtsp_flags", "prefer_tcp",
            # stimeout is not recognized by some ffmpeg builds; use timeouts below
            "-rw_timeout", "15000000",
            "-timeout", "15000000",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-max_delay", "500000",
            "-use_wallclock_as_timestamps", "1",
            "-probesize", "16M",
            "-analyzeduration", "16M",
            "-i", self.rtsp_url,
            "-an",  # drop audio
            "-r", str(self.fps),
        ]

        vf_filters = []
        # Add rotation filter
        if self.rotation_angle == 90:
            vf_filters.append("transpose=2") # Counter-clockwise 90
            # Swap dimensions if rotating by 90 or 270
            self.out_width, self.out_height = self.out_height, self.out_width
            logger.info(f"Applying 90-degree counter-clockwise rotation, new dimensions: {self.out_width}x{self.out_height}")
        elif self.rotation_angle == 180:
            vf_filters.append("transpose=2,transpose=2") # 180 (two 90 counter-clockwise)
        elif self.rotation_angle == 270:
            vf_filters.append("transpose=1") # Clockwise 90 (equivalent to counter-clockwise 270)
            # Swap dimensions if rotating by 90 or 270
            self.out_width, self.out_height = self.out_height, self.out_width
            logger.info(f"Applying 270-degree counter-clockwise rotation, new dimensions: {self.out_width}x{self.out_height}")

        # Add scaling filter
        vf_filters.append(f"scale={self.out_width}:{self.out_height}")

        if vf_filters:
            cmd += ["-vf", ",".join(vf_filters)]
            logger.info(f"FFmpeg command: {cmd}")
        
        if self.mode == 'raw':
            cmd += [
                "-pix_fmt", "bgr24",
                "-f", "rawvideo",
                "pipe:1",
            ]
        else:
            cmd += [
                "-f", "mjpeg",
                "-q:v", str(self.quality),
                "pipe:1",
            ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            # Log ffmpeg stderr in background for diagnostics
            def _log_stderr(stream):
                try:
                    while True:
                        line = stream.readline()
                        if not line:
                            break
                        logger.warning(f"ffmpeg: {line.decode(errors='ignore').strip()}")
                except Exception:
                    pass
            if self.proc.stderr is not None:
                self._stderr_thread = threading.Thread(target=_log_stderr, args=(self.proc.stderr,), daemon=True)
                self._stderr_thread.start()
            self.running = True
            self._restart_count += 1
            self._consecutive_failures = 0  # Reset failure counter on successful start
            logger.info(f"FFmpegReader started successfully (restart #{self._restart_count})")
            return True
        except Exception as e:
            logger.error(f"Failed to start ffmpeg reader: {e}")
            self._restart_count += 1
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.proc or not self.proc.stdout:
            self._consecutive_failures += 1
            return None

        try:
            if self.mode == 'raw':
                expected = self.out_width * self.out_height * 3
                # Read exactly one frame with timeout
                start_time = time.time()
                while len(self.buffer) < expected and (time.time() - start_time) < 5.0:  # 5 second timeout
                    chunk = self.proc.stdout.read(expected - len(self.buffer))
                    if not chunk:
                        if (time.time() - start_time) >= 5.0:
                            logger.warning("FFmpegReader: Timeout reading raw frame data")
                            self._consecutive_failures += 1
                            return None
                        # Check if process is still alive
                        if self.proc.poll() is not None:
                            logger.warning("FFmpegReader: FFmpeg process died while reading")
                            self._consecutive_failures += 1
                            return None
                        time.sleep(0.01)  # Small delay to prevent busy waiting
                        continue
                    self.buffer.extend(chunk)

                if len(self.buffer) < expected:
                    logger.warning("FFmpegReader: Incomplete frame data received")
                    self.buffer.clear()  # Reset buffer
                    self._consecutive_failures += 1
                    return None

                data = bytes(self.buffer[:expected])
                del self.buffer[:expected]
                frame = np.frombuffer(data, dtype=np.uint8)
                frame = frame.reshape((self.out_height, self.out_width, 3))
                self._last_successful_read = time.time()
                self._consecutive_failures = 0  # Reset failure counter on success
                return frame
            else:
                # MJPEG mode: Read chunks until we find a full JPEG frame (FFD8 ... FFD9)
                start_time = time.time()
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    if (time.time() - start_time) >= 5.0:
                        logger.warning("FFmpegReader: Timeout reading MJPEG frame data")
                        self._consecutive_failures += 1
                        return None
                    # Check if process is still alive
                    if self.proc.poll() is not None:
                        logger.warning("FFmpegReader: FFmpeg process died while reading MJPEG")
                        self._consecutive_failures += 1
                        return None
                    return None

                self.buffer.extend(chunk)
                start = self.buffer.find(b"\xff\xd8")
                end = self.buffer.find(b"\xff\xd9", start + 2)
                if start != -1 and end != -1:
                    jpeg = bytes(self.buffer[start:end + 2])
                    del self.buffer[:end + 2]
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._last_successful_read = time.time()
                        self._consecutive_failures = 0  # Reset failure counter on success
                    return frame
                return None
        except Exception as e:
            logger.error(f"FFmpegReader read error: {e}")
            self._consecutive_failures += 1
            return None

    def stop(self):
        self.running = False
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        self.buffer.clear()

    def is_healthy(self) -> bool:
        """Check if the FFmpegReader is healthy and functioning properly"""
        if not self.running or not self.proc:
            return False

        # Check if process is still alive
        if self.proc.poll() is not None:
            return False

        # Check if we haven't had too many consecutive failures
        if self._consecutive_failures >= self._max_consecutive_failures:
            return False

        # Check if we haven't exceeded max restarts
        if self._restart_count >= self._max_restarts:
            return False

        # Check if we've had a successful read recently (within last 30 seconds)
        if (time.time() - self._last_successful_read) > 30.0:
            return False

        return True

    def get_status(self) -> dict:
        """Get detailed status information about the FFmpegReader"""
        return {
            'running': self.running,
            'restart_count': self._restart_count,
            'consecutive_failures': self._consecutive_failures,
            'last_successful_read': self._last_successful_read,
            'process_alive': self.proc is not None and self.proc.poll() is None if self.proc else False,
            'buffer_size': len(self.buffer),
            'healthy': self.is_healthy()
        }

class VideoStreamHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving video stream"""

    def __init__(self, *args, streamer=None, **kwargs):
        self.streamer = streamer
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        logger.info(f"HTTP request for path: {self.path}")
        if self.path in ['/video_feed', '/cam', '/camera', '/stream', '/video']:
            # Check if client wants single image or stream
            accept_header = self.headers.get('Accept', '')
            if 'single' in self.path or (accept_header and 'image/jpeg' in accept_header):
                self.serve_single_jpeg()
            else:
                self.serve_video_feed()
        elif self.path in ['/single', '/snapshot']:
            self.serve_single_jpeg()
        elif self.path == '/health':
            self.serve_health_check()
        elif self.path == '/':
            self.serve_index()
        else:
            self.send_error(404, "Not Found")

    def serve_video_feed(self):
        """Serve MJPEG video stream"""
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        try:
            consecutive_failures = 0
            max_consecutive_failures = 50  # Allow up to 5 seconds of failures

            while self.streamer and self.streamer.is_streaming:
                # Check if streamer is healthy
                if hasattr(self.streamer, 'is_healthy') and not self.streamer.is_healthy():
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning("Video feed: Streamer unhealthy for too long, ending feed")
                        break

                    # Send a placeholder frame or wait
                    time.sleep(0.1)
                    continue
                else:
                    consecutive_failures = 0  # Reset counter if healthy

                # Get current frame
                if hasattr(self.streamer, 'get_current_frame'):
                    frame_data = self.streamer.get_current_frame()
                    if frame_data:
                        # Send frame as MJPEG
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning("Video feed: No frame data for too long, ending feed")
                            break

                time.sleep(0.1)  # 10 FPS

        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected from video feed")
        except Exception as e:
            logger.error(f"Error serving video feed: {e}")

    def serve_single_jpeg(self):
        """Serve a single JPEG image"""
        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()

        try:
            if hasattr(self.streamer, 'get_current_frame'):
                frame_data = self.streamer.get_current_frame()
                if frame_data:
                    self.wfile.write(frame_data)
                else:
                    # Check if streamer is healthy but just temporarily has no frame
                    if hasattr(self.streamer, 'is_healthy') and self.streamer.is_healthy():
                        logger.warning("No frame data available for single JPEG but streamer is healthy - waiting briefly")
                        # Wait a short time and try once more
                        time.sleep(0.5)
                        frame_data = self.streamer.get_current_frame()
                        if frame_data:
                            self.wfile.write(frame_data)
                        else:
                            logger.warning("Still no frame data available for single JPEG")
                            # Send a minimal placeholder response
                            self.wfile.write(b'')
                    else:
                        logger.warning("No frame data available for single JPEG and streamer unhealthy")
                        # Send a minimal placeholder response
                        self.wfile.write(b'')
            else:
                logger.error("Streamer does not have get_current_frame method")
                # Send a minimal placeholder response
                self.wfile.write(b'')
        except Exception as e:
            logger.error(f"Error serving single JPEG: {e}")

    def serve_health_check(self):
        """Serve health check endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()

        try:
            import json
            if hasattr(self.streamer, 'get_health_status'):
                health_data = self.streamer.get_health_status()
                self.wfile.write(json.dumps(health_data, indent=2).encode())
            else:
                # Basic health check if detailed status not available
                basic_health = {
                    'streaming': self.streamer.is_streaming if self.streamer else False,
                    'status': 'ok' if (self.streamer and self.streamer.is_streaming) else 'degraded'
                }
                self.wfile.write(json.dumps(basic_health, indent=2).encode())
        except Exception as e:
            logger.error(f"Error serving health check: {e}")
            error_response = {'error': str(e), 'status': 'error'}
            self.wfile.write(json.dumps(error_response, indent=2).encode())

    def serve_index(self):
        """Serve simple HTML page"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Get the actual IP address for LAN access
        try:
            local_ip = _get_local_ip()
            if local_ip:
                ip_info = f" or http://{local_ip}:{self.server.server_address[1]}"
            else:
                ip_info = ""
        except:
            ip_info = ""

        html = f"""
        <html>
        <head><title>Eufy Camera Stream</title></head>
        <body>
        <h1>Eufy Security Camera Stream</h1>
        <p>Server running at: http://localhost:{self.server.server_address[1]}{ip_info}</p>
        <p>Available endpoints:</p>
        <ul>
        <li><a href="/video_feed">MJPEG Stream (video_feed)</a></li>
        <li><a href="/cam">MJPEG Stream (cam)</a></li>
        <li><a href="/camera">MJPEG Stream (camera)</a></li>
        <li><a href="/stream">MJPEG Stream (stream)</a></li>
        <li><a href="/video">MJPEG Stream (video)</a></li>
        <li><a href="/single">Single JPEG Image (single)</a></li>
        <li><a href="/snapshot">Single JPEG Image (snapshot)</a></li>
        <li><a href="/health">Health Check (JSON)</a></li>
        </ul>
        <img src="/video_feed" width="800" height="600" />
        </body>
        </html>
        """
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"HTTP {self.address_string()} - {format % args}")


class VideoStreamServer:
    """HTTP server for serving video stream"""

    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.streamer = None

    def set_streamer(self, streamer):
        """Set the streamer instance for frame access"""
        self.streamer = streamer

    def start(self):
        """Start the HTTP server"""
        try:
            # Create custom handler class with streamer reference
            def handler(*args, **kwargs):
                return VideoStreamHandler(*args, streamer=self.streamer, **kwargs)

            # Try base port, then incrementally search for a free one
            base_port = self.port
            for candidate in range(base_port, base_port + 20):
                try:
                    self.server = HTTPServer((self.host, candidate), handler)
                    self.port = candidate
                    # Get the actual IP address for LAN access
                    try:
                        local_ip = _get_local_ip()
                        if local_ip:
                            logger.info(f"HTTP server started at http://localhost:{self.port} or http://{local_ip}:{self.port}")
                        else:
                            logger.info(f"HTTP server started at http://localhost:{self.port} (bound to all interfaces)")
                    except:
                        logger.info(f"HTTP server started at http://localhost:{self.port} (bound to all interfaces)")
                    server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                    server_thread.start()
                    return True
                except OSError as e:
                    # 48 = EADDRINUSE on macOS, 98 on Linux
                    if getattr(e, 'errno', None) in (48, 98):
                        continue
                    raise
            logger.error(f"Failed to start HTTP server: no free port in range {base_port}-{base_port+19}")
            return False
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            return False

    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("HTTP server stopped")


class EufyRTSPStreamer:
    """
    Class to handle RTSP streaming from Eufy security cameras
    """

    def __init__(self, rtsp_url: str, window_width: int = 800, window_height: int = 600, http_port: int = 8080, headless: bool = False, rotation_angle: int = 0):
        """
        Initialize the RTSP streamer with HTTP server

        Args:
            rtsp_url: Full RTSP URL to the camera stream
            window_width: Display window width
            window_height: Display window height
            http_port: Port for HTTP video server
        """
        self.rtsp_url = rtsp_url
        self.window_width = window_width
        self.window_height = window_height
        self.http_port = http_port
        self.cap = None
        self.is_streaming = False
        self.stream_thread = None
        self.http_server = None
        self.headless = headless
        self.rotation_angle = rotation_angle # Store rotation angle

        # Frame buffer for HTTP serving
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.backend = 'opencv'
        self.ffmpeg_reader = None

        # Initialize HTTP server
        self.http_server = VideoStreamServer(port=http_port)
        self.http_server.set_streamer(self)

        # GUI only when not headless
        if not self.headless:
            self.root = tk.Tk()
            self.root.title("Eufy Security Camera Stream")
            # Adjust window dimensions if rotation is applied
            if self.rotation_angle == 90 or self.rotation_angle == 270:
                self.root.geometry(f"{window_height}x{window_width + 50}")
            else:
                self.root.geometry(f"{window_width}x{window_height + 50}")

            # Create video display label
            self.video_label = ttk.Label(self.root)
            self.video_label.pack(padx=10, pady=10)

            # Control buttons
            self.control_frame = ttk.Frame(self.root)
            self.control_frame.pack(fill=tk.X, padx=10, pady=5)

            self.start_button = ttk.Button(
                self.control_frame,
                text="Start Stream",
                command=self.start_stream
            )
            self.start_button.pack(side=tk.LEFT, padx=5)

            self.stop_button = ttk.Button(
                self.control_frame,
                text="Stop Stream",
                command=self.stop_stream,
                state=tk.DISABLED
            )
            self.stop_button.pack(side=tk.LEFT, padx=5)

            # Status label
            self.status_var = tk.StringVar()
            self.status_var.set("Ready to connect")
            self.status_label = ttk.Label(
                self.control_frame,
                textvariable=self.status_var
            )
            self.status_label.pack(side=tk.RIGHT, padx=10)
        else:
            # Headless placeholders
            self.root = None
            self.video_label = None
            self.control_frame = None
            self.start_button = None
            self.stop_button = None
            self.status_var = None
            self.status_label = None

    def load_env_config(self) -> dict:
        """
        Load configuration from .env file

        Returns:
            Dictionary containing configuration parameters
        """
        config = {}

        try:
            # Try to load from .env file
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
            else:
                logger.warning(".env file not found. Using default configuration.")
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")

        return config

    def build_rtsp_url(self, config: dict) -> str:
        """
        Build RTSP URL from configuration

        Args:
            config: Configuration dictionary

        Returns:
            Complete RTSP URL
        """
        # If full RTSP URL is provided, normalize it to enforce TCP and /live0
        if 'CAMERA_RTSP_URL' in config and config['CAMERA_RTSP_URL'].startswith('rtsp://'):
            raw = config['CAMERA_RTSP_URL']
            try:
                parsed = urllib.parse.urlsplit(raw)
                username = parsed.username or ''
                password = parsed.password or ''
                host = parsed.hostname or config.get('CAMERA_IP', '192.168.1.100')
                port = parsed.port or int(config.get('CAMERA_PORT', '554'))
                # Lock to /live0 for Eufy Indoor Cam 2K
                path = '/live0'
                userinfo = ''
                if username or password:
                    userinfo = f"{_encode_userinfo(username)}:{_encode_userinfo(password)}@"
                url = f"rtsp://{userinfo}{host}:{port}{path}"
                logger.info(f"Built RTSP URL (normalized): {_redact_rtsp(url)}")
                return url
            except Exception:
                # Fallback to provided string
                logger.info(f"Using provided RTSP URL: {_redact_rtsp(raw)}")
                return raw

        # Build RTSP URL from components
        username = config.get('CAMERA_USERNAME', 'admin')
        password = config.get('CAMERA_PASSWORD', '')
        ip = config.get('CAMERA_IP', '192.168.1.100')
        port = config.get('CAMERA_PORT', '554')
        # Lock to /live0
        userinfo = f"{_encode_userinfo(username)}:{_encode_userinfo(password)}@" if username or password else ''
        rtsp_url = f"rtsp://{userinfo}{ip}:{port}/live0"
        logger.info(f"Built RTSP URL: {_redact_rtsp(rtsp_url)}")
        return rtsp_url

    def _healthcheck_rtsp(self, url: str) -> bool:
        """Run a lightweight ffprobe to validate transport/path/auth."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-rtsp_transport", "tcp",
                "-timeout", "5000000",
                "-show_streams",
                url,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if proc.returncode == 0:
                return True
            err = proc.stderr.decode(errors='ignore')
            if '401' in err:
                logger.error("RTSP auth failed (401 Unauthorized)")
                return False
            if '404' in err:
                logger.error("RTSP path not found (404). Ensure /live0 is correct.")
                return False
            if '461' in err:
                logger.error("RTSP transport unsupported (461). Forcing TCP.")
                return False
            # Treat common network errors as fatal
            fatal_net_signatures = [
                'No route to host',
                'Network is unreachable',
                'Connection refused',
                'Connection timed out',
                'timed out',
                'ECONNREFUSED',
                'EHOSTUNREACH',
            ]
            if any(sig in err for sig in fatal_net_signatures):
                logger.error(f"RTSP network error: {err.strip()}")
                return False
            # Default: treat unknown errors as fatal to avoid confusing retries
            if err.strip():
                logger.error(f"RTSP healthcheck failed: {err.strip()}")
            return False
        except FileNotFoundError:
            # ffprobe not available; skip healthcheck
            logger.warning("ffprobe not found; skipping RTSP healthcheck")
            return True
        except Exception as e:
            logger.error(f"Healthcheck error: {e}")
            return False

    def initialize_capture(self) -> bool:
        """
        Initialize video capture from RTSP stream

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Connecting to RTSP stream (OpenCV): {_redact_rtsp(self.rtsp_url)}")

            # Optional healthcheck
            url_candidates = [self.rtsp_url]
            # If normalized to /live0, build fallbacks for /live1 and /live
            try:
                if urllib.parse.urlsplit(self.rtsp_url).path == '/live0':
                    url_candidates.append(_replace_rtsp_path(self.rtsp_url, '/live1'))
                    url_candidates.append(_replace_rtsp_path(self.rtsp_url, '/live'))
            except Exception:
                pass

            healthy_url = None
            for candidate in url_candidates:
                ok = self._healthcheck_rtsp(candidate)
                if ok:
                    healthy_url = candidate
                    break
                else:
                    # If 404 was logged, try next candidate
                    continue
            if not healthy_url:
                if self.status_var is not None:
                    self.status_var.set("RTSP healthcheck failed")
                return False
            if healthy_url != self.rtsp_url:
                logger.info(f"RTSP path fallback selected: {_redact_rtsp(healthy_url)}")
                self.rtsp_url = healthy_url

            # Force TCP for OpenCV FFmpeg backend
            try:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            except Exception:
                pass

            # Prefer FFmpeg backend
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                # Test if we can read a frame quickly
                ok, test_frame = self.cap.read()
                if ok and test_frame is not None:
                    logger.info("Connected via OpenCV/FFmpeg backend")
                    if self.status_var is not None:
                        self.status_var.set("Connected (OpenCV)")
                    self.backend = 'opencv'
                    return True
                else:
                    logger.warning("OpenCV opened but failed to deliver frame; falling back to ffmpeg pipe")

            # Fallback to FFmpegReader
            # Try raw BGR pipe first to avoid MJPEG framing dependency
            # Pass rotation_angle to FFmpegReader
            self.ffmpeg_reader = FFmpegReader(self.rtsp_url, mode='raw', out_width=self.window_width, out_height=self.window_height, rotation_angle=self.rotation_angle)
            if self.ffmpeg_reader.start():
                # Try to get an initial frame (allow more time for codec params)
                start = time.time()
                frame = None
                while time.time() - start < 10.0 and frame is None:
                    frame = self.ffmpeg_reader.read_frame()
                if frame is not None:
                    logger.info("Connected via ffmpeg subprocess reader")
                    if self.status_var is not None:
                        self.status_var.set("Connected (FFmpeg)")
                    self.backend = 'ffmpeg'
                    # Seed current_frame buffer (no color conversion)
                    with self.frame_lock:
                        _, jpeg_frame = cv2.imencode('.jpg', frame)
                        self.current_frame = jpeg_frame.tobytes()
                    return True
            logger.error("Failed to open RTSP stream with both OpenCV and FFmpeg")
            if self.status_var is not None:
                self.status_var.set("Failed to connect to camera")
                return False

        except Exception as e:
            logger.error(f"Error initializing capture: {e}")
            if self.status_var is not None:
                self.status_var.set(f"Connection error: {str(e)}")
            return False

    def stream_video(self):
        """Main streaming loop running in separate thread"""
        try:
            backoff = 1.0
            max_backoff = 30.0
            max_reconnect_attempts = 50
            reconnect_attempts = 0
            circuit_breaker_failures = 0
            circuit_breaker_threshold = 5
            last_successful_frame = time.time()

            while self.is_streaming:
                frame = None
                backend_healthy = True

                try:
                    if getattr(self, 'backend', 'opencv') == 'opencv':
                        if not (self.cap and self.cap.isOpened()):
                            raise RuntimeError("OpenCV capture closed")
                        ok, frame = self.cap.read()
                        if not ok:
                            frame = None
                    else:
                        if not self.ffmpeg_reader:
                            raise RuntimeError("FFmpeg reader missing")
                        frame = self.ffmpeg_reader.read_frame()
                        # Check FFmpegReader health
                        if not self.ffmpeg_reader.is_healthy():
                            backend_healthy = False

                except Exception as e:
                    logger.error(f"Backend error during frame read: {e}")
                    frame = None
                    backend_healthy = False

                if frame is None or not backend_healthy:
                    # Check if this is a circuit breaker situation
                    if frame is None:
                        circuit_breaker_failures += 1
                    else:
                        circuit_breaker_failures = 0  # Reset if we got a frame but backend unhealthy

                    if circuit_breaker_failures >= circuit_breaker_threshold:
                        logger.warning(f"Circuit breaker triggered after {circuit_breaker_failures} failures")
                        if self.status_var is not None:
                            self.status_var.set("Circuit breaker - waiting before reconnect...")

                        # Wait longer before attempting reconnect
                        circuit_break_wait = min(10.0, backoff * 2)
                        time.sleep(circuit_break_wait)

                        # Reset circuit breaker counter
                        circuit_breaker_failures = 0

                    # Check if we've exceeded max reconnect attempts
                    if reconnect_attempts >= max_reconnect_attempts:
                        logger.error(f"Exceeded maximum reconnect attempts ({max_reconnect_attempts})")
                        if self.status_var is not None:
                            self.status_var.set("Max reconnect attempts exceeded")
                        time.sleep(5.0)  # Wait before potentially trying again
                        reconnect_attempts = 0  # Reset for potential recovery
                        continue

                    logger.warning(f"Frame read failed (attempt {reconnect_attempts + 1}); attempting reconnect with backoff")
                    if self.status_var is not None:
                        self.status_var.set(f"Reconnecting... (attempt {reconnect_attempts + 1})")

                    # Exponential backoff with jitter
                    jitter = np.random.uniform(0.8, 1.2)
                    sleep_time = min(backoff * jitter, max_backoff)
                    time.sleep(sleep_time)

                    # Reinitialize
                    try:
                        # Stop existing backends
                        if self.cap and self.cap.isOpened():
                            self.cap.release()
                        if self.ffmpeg_reader:
                            self.ffmpeg_reader.stop()
                    except Exception as e:
                        logger.warning(f"Error stopping backends during reconnect: {e}")

                    if self.initialize_capture():
                        backoff = max(1.0, backoff * 0.8)  # Reduce backoff on successful reconnect
                        reconnect_attempts = 0
                        logger.info("Successfully reconnected to stream")
                        if self.status_var is not None:
                            self.status_var.set("Reconnected")
                    else:
                        backoff = min(backoff * 1.5, max_backoff)
                        reconnect_attempts += 1
                        logger.warning(f"Failed to reconnect (attempt {reconnect_attempts})")
                    continue

                # Reset circuit breaker failures on successful frame
                circuit_breaker_failures = 0
                reconnect_attempts = 0
                last_successful_frame = time.time()

                # Apply rotation if needed (OpenCV backend doesn't have FFmpeg filters)
                if self.rotation_angle != 0 and getattr(self, 'backend', 'opencv') == 'opencv':
                    if self.rotation_angle == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self.rotation_angle == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif self.rotation_angle == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Update frame buffer for HTTP serving (no color conversion)
                with self.frame_lock:
                    # Use frame as-is for HTTP serving
                    _, jpeg_frame = cv2.imencode('.jpg', frame)
                    self.current_frame = jpeg_frame.tobytes()

                # GUI update only when not headless
                if not self.headless and self.root is not None:
                    # Resize frame for display
                    # Adjust display dimensions if rotation is applied
                    if self.rotation_angle == 90 or self.rotation_angle == 270:
                        display_frame = cv2.resize(frame, (self.window_height, self.window_width))
                    else:
                        display_frame = cv2.resize(frame, (self.window_width, self.window_height))
                    rgb_frame = display_frame
                    photo = tk.PhotoImage(data=cv2.imencode('.ppm', rgb_frame)[1].tobytes())
                    self.root.after(0, self.update_video_display, photo)

                # Small delay to prevent overwhelming the system
                time.sleep(0.033)  # ~30 FPS

        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if getattr(self, 'ffmpeg_reader', None):
                self.ffmpeg_reader.stop()
            logger.info("Streaming stopped")

    def get_current_frame(self):
        """Get current frame data for HTTP serving"""
        with self.frame_lock:
            return self.current_frame

    def is_healthy(self) -> bool:
        """Check if the streaming system is healthy"""
        if not self.is_streaming:
            return False

        # Check if we have a current frame
        if self.current_frame is None:
            return False

        # Check backend health
        if hasattr(self, 'backend'):
            if self.backend == 'ffmpeg' and self.ffmpeg_reader:
                if not self.ffmpeg_reader.is_healthy():
                    return False
            elif self.backend == 'opencv' and self.cap:
                if not self.cap.isOpened():
                    return False

        # Check if we've had recent successful streaming activity
        if hasattr(self, 'stream_thread') and self.stream_thread:
            if not self.stream_thread.is_alive():
                return False

        return True

    def get_health_status(self) -> dict:
        """Get comprehensive health status of the streaming system"""
        status = {
            'streaming': self.is_streaming,
            'healthy': self.is_healthy(),
            'backend': getattr(self, 'backend', None),
            'current_frame_available': self.current_frame is not None,
            'stream_thread_alive': False,
            'last_frame_time': None
        }

        if hasattr(self, 'stream_thread') and self.stream_thread:
            status['stream_thread_alive'] = self.stream_thread.is_alive()

        # Get backend-specific status
        if self.backend == 'ffmpeg' and self.ffmpeg_reader:
            status['ffmpeg_status'] = self.ffmpeg_reader.get_status()
        elif self.backend == 'opencv' and self.cap:
            status['opencv_open'] = self.cap.isOpened()

        return status

    def start_heartbeat_monitor(self):
        """Start a background thread to monitor connection health"""
        def heartbeat_worker():
            while self.is_streaming:
                try:
                    if not self.is_healthy():
                        logger.warning("Heartbeat: Streaming system unhealthy, logging status")
                        health_status = self.get_health_status()
                        logger.info(f"Health status: {health_status}")

                        # If consistently unhealthy for too long, attempt recovery
                        if hasattr(self, '_heartbeat_failures'):
                            self._heartbeat_failures += 1
                        else:
                            self._heartbeat_failures = 1

                        if self._heartbeat_failures >= 3:
                            logger.warning("Heartbeat: Multiple failures detected, triggering recovery")
                            if self.status_var is not None:
                                self.status_var.set("Health check failed - recovering...")
                            # The recovery will be handled by the main streaming loop
                            self._heartbeat_failures = 0
                    else:
                        self._heartbeat_failures = 0

                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")

                time.sleep(10.0)  # Check every 10 seconds

        heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        heartbeat_thread.start()
        logger.info("Heartbeat monitor started")

    def update_video_display(self, photo):
        """Update the video display in the GUI (called from main thread)"""
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Keep reference

    def start_stream(self):
        """Start the RTSP streaming"""
        if self.is_streaming:
            logger.warning("Stream already running")
            return

        # Load configuration
        config = self.load_env_config()
        self.rtsp_url = self.build_rtsp_url(config)

        # Update window size from config if available
        if 'WINDOW_WIDTH' in config:
            self.window_width = int(config['WINDOW_WIDTH'])
        if 'WINDOW_HEIGHT' in config:
            self.window_height = int(config['WINDOW_HEIGHT'])

        # Update HTTP port from config if available
        if 'HTTP_PORT' in config:
            self.http_port = int(config['HTTP_PORT'])

        if not self.initialize_capture():
            if not self.headless:
                messagebox.showerror("Connection Error", "Failed to connect to camera stream")
            else:
                logger.error("Failed to connect to camera stream")
            return

        # Start HTTP server
        if not self.http_server.start():
            if not self.headless:
                messagebox.showerror("HTTP Server Error", "Failed to start HTTP server")
            else:
                logger.error("Failed to start HTTP server")
            return

        self.is_streaming = True
        if not self.headless and self.start_button is not None and self.stop_button is not None:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

        # Start heartbeat monitor
        self.start_heartbeat_monitor()

        # Start streaming in separate thread
        self.stream_thread = threading.Thread(target=self.stream_video, daemon=True)
        self.stream_thread.start()

        # Get the actual IP address for LAN access
        try:
            local_ip = _get_local_ip()
            if local_ip:
                logger.info(f"RTSP streaming started with HTTP server at http://localhost:{self.http_server.port} or http://{local_ip}:{self.http_server.port}")
                logger.info(f"Access video feed at: http://localhost:{self.http_server.port}/video_feed or http://{local_ip}:{self.http_server.port}/video_feed")
            else:
                logger.info(f"RTSP streaming started with HTTP server at http://localhost:{self.http_server.port}")
                logger.info(f"Access video feed at: http://localhost:{self.http_server.port}/video_feed")
        except:
            logger.info(f"RTSP streaming started with HTTP server at http://localhost:{self.http_server.port}")
            logger.info(f"Access video feed at: http://localhost:{self.http_server.port}/video_feed")

    def start_headless(self):
        """Start streaming without GUI components."""
        if self.is_streaming:
            logger.warning("Stream already running")
            return False

        # Load configuration
        config = self.load_env_config()
        self.rtsp_url = self.build_rtsp_url(config)

        # Update HTTP port from config if available
        if 'HTTP_PORT' in config:
            self.http_port = int(config['HTTP_PORT'])

        # Initialize capture
        if not self.initialize_capture():
            logger.error("Failed to connect to camera stream (headless)")
            return False

        # Start HTTP server
        if not self.http_server.start():
            logger.error("Failed to start HTTP server (headless)")
            return False

        self.is_streaming = True

        # Start heartbeat monitor
        self.start_heartbeat_monitor()

        # Start streaming in separate thread
        self.stream_thread = threading.Thread(target=self.stream_video, daemon=True)
        self.stream_thread.start()

        try:
            local_ip = _get_local_ip()
            if local_ip:
                logger.info(f"Headless RTSP streaming with HTTP server at http://localhost:{self.http_server.port} or http://{local_ip}:{self.http_server.port}")
            else:
                logger.info(f"Headless RTSP streaming with HTTP server at http://localhost:{self.http_server.port}")
        except:
            logger.info(f"Headless RTSP streaming with HTTP server at http://localhost:{self.http_server.port}")
        return True

    def stop_stream(self):
        """Stop the RTSP streaming"""
        if not self.is_streaming:
            return

        logger.info("Stopping RTSP stream...")
        self.is_streaming = False

        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)

        if self.cap and self.cap.isOpened():
            self.cap.release()

        if not self.headless and self.video_label is not None:
            self.video_label.config(image='')
        if not self.headless and self.start_button is not None and self.stop_button is not None:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        if self.status_var is not None:
            self.status_var.set("Stream stopped")

        # Stop HTTP server
        self.http_server.stop()

        logger.info("RTSP streaming stopped")

    def run(self):
        """Run the GUI application"""
        try:
            if self.root is not None:
                self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
                self.root.mainloop()
        except Exception as e:
            logger.error(f"Error running GUI: {e}")

    def on_closing(self):
        """Handle window closing event"""
        self.stop_stream()
        self.root.destroy()


def main():
    """Main function"""
    logger.info("Starting Eufy RTSP Streamer")

    # Load configuration from .env file
    config = {}
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()

    # Build RTSP URL
    rtsp_url = ""
    if 'CAMERA_RTSP_URL' in config:
        rtsp_url = config['CAMERA_RTSP_URL']

    if not rtsp_url:
        logger.error("No RTSP URL found in .env file")
        print("Please create a .env file with your camera configuration:")
        print("CAMERA_RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream")
        print("Or configure individual components:")
        print("CAMERA_USERNAME=admin")
        print("CAMERA_PASSWORD=your_password")
        print("CAMERA_IP=192.168.1.100")
        print("CAMERA_PORT=554")
        print("STREAM_PATH=stream")
        sys.exit(1)

    # Get display dimensions
    window_width = int(config.get('WINDOW_WIDTH', '800'))
    window_height = int(config.get('WINDOW_HEIGHT', '600'))

    # Get HTTP server configuration
    http_port = int(config.get('HTTP_PORT', '8080'))

    # Get rotation angle from config
    rotation_angle = int(config.get('ROTATION_ANGLE', '0'))

    # Create and run streamer
    streamer = EufyRTSPStreamer(rtsp_url, window_width, window_height, http_port, rotation_angle=rotation_angle)
    streamer.run()


if __name__ == "__main__":
    main()
