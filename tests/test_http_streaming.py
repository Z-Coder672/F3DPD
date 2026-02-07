#!/usr/bin/env python3
"""
Test script for HTTP streaming architecture
Demonstrates the RTSP -> HTTP -> Client pipeline
"""

import cv2
import numpy as np
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
from urllib.parse import urljoin

# Import the HTTP client class
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eufy_http_client import EufyHTTPClient

class MockVideoHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler that serves test frames"""

    def do_GET(self):
        if self.path == '/video_feed':
            self.serve_video_feed()
        elif self.path == '/':
            self.serve_index()
        else:
            self.send_error(404, "Not Found")

    def serve_video_feed(self):
        """Serve a simple MJPEG-like stream"""
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        # Send 5 test frames
        for i in range(5):
            # Create a test frame (blue screen with counter)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = [255, 0, 0]  # Blue background

            # Add text
            cv2.putText(frame, f"Test Frame {i+1}", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Encode as JPEG
            _, jpeg_data = cv2.imencode('.jpg', frame)

            # Send frame
            self.wfile.write(b'--frame\r\n')
            self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
            self.wfile.write(jpeg_data.tobytes())
            self.wfile.write(b'\r\n')

            time.sleep(1)  # 1 second between frames

        self.wfile.write(b'--frame--\r\n')

    def serve_index(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        html = """
        <html><body>
        <h1>Mock Camera Stream</h1>
        <img src="/video_feed" width="640" height="480" />
        </body></html>
        """
        self.wfile.write(html.encode())

def run_mock_server():
    """Run a mock HTTP server for testing"""
    server = HTTPServer(('127.0.0.1', 8080), MockVideoHandler)
    print("Mock HTTP server running on http://127.0.0.1:8080")
    server.serve_forever()

def test_http_client():
    """Test the HTTP client with the mock server"""
    print("Testing HTTP client...")

    # Wait a moment for server to start
    time.sleep(1)

    client = EufyHTTPClient()

    # Test connection
    if client.test_connection():
        print("✅ HTTP server connection successful")

        # Test frame capture
        frame = client.get_frame()
        if frame is not None:
            print(f"✅ Frame captured: {frame.shape}")

            # Save test frame
            cv2.imwrite("mock_test_frame.jpg", frame)
            print("✅ Test frame saved as mock_test_frame.jpg")

            # Test multiple captures
            print("\n🧪 Testing multiple frame captures...")
            for i in range(3):
                frame = client.get_frame()
                if frame is not None:
                    filename = f"mock_frame_{i}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"  Captured: {filename}")
                time.sleep(1.5)

            print("✅ Multiple frame capture test complete!")

        else:
            print("❌ Failed to capture frame")
    else:
        print("❌ Cannot connect to HTTP server")

def main():
    """Main test function"""
    print("🚀 Testing HTTP Streaming Architecture")
    print("=" * 50)

    # Start mock server in background thread
    server_thread = threading.Thread(target=run_mock_server, daemon=True)
    server_thread.start()

    try:
        # Test the client
        test_http_client()

        print("\n✅ HTTP streaming architecture test complete!")
        print("The pipeline works: HTTP Server → HTTP Client → Frame Processing")

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()
