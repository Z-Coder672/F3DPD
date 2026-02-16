#!/usr/bin/env python3
"""
Send gcode commands to the printer over TCP and parse responses.
"""

import logging
import re
import subprocess
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

PRINTER_HOST = "192.168.50.20"
PRINTER_PORT = "8899"
GCODE_TIMEOUT = 30


def _send_gcode(command: str, read_after_ok: float = 0) -> Optional[str]:
    try:
        proc = subprocess.Popen(
            ["nc", PRINTER_HOST, PRINTER_PORT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.error("Failed to start nc for %s: %s", command, e)
        return None

    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(f"~{command}\r\n".encode())
        proc.stdin.flush()

        chunks = []
        deadline = time.monotonic() + GCODE_TIMEOUT
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            chunks.append(decoded)
            if decoded.strip() == "ok":
                if read_after_ok > 0:
                    deadline = min(deadline, time.monotonic() + read_after_ok)
                else:
                    break

        output = "".join(chunks)
        return output if output else None
    except Exception as e:
        logger.error("Error reading gcode response for %s: %s", command, e)
        return None
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.stdin.close()
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


def send_pause() -> Optional[str]:
    return _send_gcode("M25")


def send_resume() -> Optional[str]:
    return _send_gcode("M24")


def send_m119() -> Optional[str]:
    return _send_gcode("M119")


def send_m27() -> Optional[str]:
    return _send_gcode("M27")


def parse_m119(output: str) -> Optional[dict]:
    machine_match = re.search(r"MachineStatus:\s*(\S+)", output)
    move_match = re.search(r"MoveMode:\s*(\S+)", output)
    current_file_match = re.search(r"CurrentFile:\s*(.*)", output)
    if not machine_match or not move_match:
        return None
    current_file = current_file_match.group(1).strip() if current_file_match else ""
    return {
        "machine_status": machine_match.group(1),
        "move_mode": move_match.group(1),
        "current_file": current_file,
    }


def parse_m27(output: str) -> Optional[dict]:
    byte_match = re.search(r"SD printing byte (\d+)/(\d+)", output)
    layer_match = re.search(r"Layer:\s*(\d+)/(\d+)", output)
    if not byte_match or not layer_match:
        return None
    return {
        "byte_current": int(byte_match.group(1)),
        "byte_total": int(byte_match.group(2)),
        "layer_current": int(layer_match.group(1)),
        "layer_total": int(layer_match.group(2)),
    }


def send_m661() -> Optional[str]:
    return _send_gcode("M661", read_after_ok=5)


def send_m23(filepath: str) -> Optional[str]:
    return _send_gcode(f"M23 {filepath}", read_after_ok=5)


def parse_m661(output: str) -> List[str]:
    matches = re.findall(r"/data/([^:]+?\.gcode)", output)
    return matches


def parse_m23(output: str) -> Optional[dict]:
    match = re.search(r"File opened:\s*(.+?)\s*/\s*Size:\s*(\d+)", output)
    if not match:
        return None
    return {
        "filename": match.group(1).strip(),
        "size": int(match.group(2)),
    }
