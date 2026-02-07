#!/usr/bin/env python3
"""
Test script for Eufy RTSP Streamer
Verifies that the program can be imported and basic functionality works
"""

import os
import sys
import logging

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import cv2
        print("✓ OpenCV (cv2) imported successfully")

        import tkinter as tk
        from tkinter import ttk
        print("✓ Tkinter imported successfully")

        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_env_file():
    """Test .env file loading"""
    try:
        from dotenv import load_dotenv

        # Try to load .env file
        load_dotenv()

        # Check if we can read environment variables
        rtsp_url = os.getenv('CAMERA_RTSP_URL')
        camera_ip = os.getenv('CAMERA_IP')

        if rtsp_url or camera_ip:
            print("✓ .env file loaded successfully")
            if rtsp_url:
                print(f"  RTSP URL: {rtsp_url.replace(rtsp_url.split(':')[2].split('@')[0], '***:***')}")
            if camera_ip:
                print(f"  Camera IP: {camera_ip}")
            return True
        else:
            print("! .env file found but no camera configuration detected")
            print("  Please configure your camera settings in .env file")
            return False

    except Exception as e:
        print(f"✗ Error loading .env file: {e}")
        return False

def test_program_import():
    """Test that the main program can be imported"""
    try:
        # Add current directory to path for imports
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        import eufy_rtsp_streamer
        print("✓ Main program imported successfully")

        # Test class instantiation (without GUI)
        streamer = eufy_rtsp_streamer.EufyRTSPStreamer("rtsp://test:test@127.0.0.1:554/test", 400, 300)
        print("✓ EufyRTSPStreamer class instantiated successfully")

        return True
    except ImportError as e:
        print(f"✗ Error importing main program: {e}")
        return False
    except Exception as e:
        print(f"✗ Error instantiating class: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Eufy RTSP Streamer...")
    print("=" * 40)

    tests = [
        ("Import Test", test_imports),
        ("Environment File Test", test_env_file),
        ("Program Import Test", test_program_import),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        results.append(test_func())

    print("\n" + "=" * 40)
    print("Test Summary:")

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The RTSP streamer should work correctly.")
        print("\nTo run the program:")
        print("  python eufy_rtsp_streamer.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        if not results[0]:  # Import test failed
            print("\nTroubleshooting:")
            print("1. Install missing dependencies: pip install opencv-python python-dotenv")
            print("2. Make sure you're using Python 3.7+")
        elif not results[1]:  # Env file test failed
            print("\nTroubleshooting:")
            print("1. Create a .env file in the project directory")
            print("2. Add your camera configuration to the .env file")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



