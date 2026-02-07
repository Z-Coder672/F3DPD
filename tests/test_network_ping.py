#!/usr/bin/env python3
"""
Network connectivity test:
- Detect the USB Wi-Fi interface (TP-Link adapter typically shows up as USB)
- Avoid the onboard Wi-Fi if it's running as an AP (broadcasting `EufyJail`)
- Ping 8.8.8.8 *through the selected interface* using `ping -I <iface>`

This is an integration/health-check style test script (similar to other `test_*.py` files here).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Optional


PING_TARGET = "8.8.8.8"
AP_SSID_NAME = "EufyJail"


@dataclass(frozen=True)
class WifiIface:
    name: str
    dev_path: Optional[str]  # realpath of /sys/class/net/<iface>/device
    is_usb: bool
    iw_type: Optional[str]  # "AP", "managed", etc.


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )


def _sysfs_ifaces() -> Iterable[str]:
    try:
        return sorted(os.listdir("/sys/class/net"))
    except FileNotFoundError:
        return []


def _is_wireless_iface(iface: str) -> bool:
    # A reliable check: wireless interfaces expose /sys/class/net/<iface>/wireless
    return os.path.isdir(f"/sys/class/net/{iface}/wireless")


def _iface_dev_realpath(iface: str) -> Optional[str]:
    dev_link = f"/sys/class/net/{iface}/device"
    if not os.path.exists(dev_link):
        return None
    try:
        return os.path.realpath(dev_link)
    except OSError:
        return None


def _is_usb_dev_path(dev_path: Optional[str]) -> bool:
    if not dev_path:
        return False
    # Commonly contains ".../usbX/..." for USB NICs.
    return "/usb" in dev_path


def _iw_iface_type(iface: str) -> Optional[str]:
    if not shutil.which("iw"):
        return None
    cp = _run(["iw", "dev", iface, "info"])
    if cp.returncode != 0:
        return None
    # Example line: "type managed" / "type AP"
    m = re.search(r"^\s*type\s+(\S+)\s*$", cp.stdout, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1)


def _is_ap_iface(iface: WifiIface) -> bool:
    # Primary: iw-reported type
    if iface.iw_type:
        return iface.iw_type.lower() == "ap"
    return False


def _ip_link_is_up(iface: str) -> Optional[bool]:
    if not shutil.which("ip"):
        return None
    cp = _run(["ip", "link", "show", "dev", iface])
    if cp.returncode != 0:
        return None
    # Look for "state UP" or flags containing "UP"
    if "state UP" in cp.stdout:
        return True
    # If we can see the line but it's not UP, consider it down-ish.
    return False


def _ipv4_addrs(iface: str) -> list[str]:
    if not shutil.which("ip"):
        return []
    cp = _run(["ip", "-4", "addr", "show", "dev", iface])
    if cp.returncode != 0:
        return []
    addrs: list[str] = []
    for line in cp.stdout.splitlines():
        line = line.strip()
        # Example: "inet 192.168.1.10/24 brd 192.168.1.255 scope global dynamic wlan1"
        if line.startswith("inet "):
            addrs.append(line.split()[1])
    return addrs


def discover_wifi_ifaces() -> list[WifiIface]:
    out: list[WifiIface] = []
    for iface in _sysfs_ifaces():
        if iface == "lo":
            continue
        if not _is_wireless_iface(iface):
            continue
        dev_path = _iface_dev_realpath(iface)
        out.append(
            WifiIface(
                name=iface,
                dev_path=dev_path,
                is_usb=_is_usb_dev_path(dev_path),
                iw_type=_iw_iface_type(iface),
            )
        )
    return out


def pick_usb_wifi_iface(ifaces: list[WifiIface]) -> Optional[WifiIface]:
    # Prefer: USB wifi that is NOT an AP
    candidates = [i for i in ifaces if i.is_usb and not _is_ap_iface(i)]
    if candidates:
        return candidates[0]

    # Fallback: any wifi that is NOT an AP (better than nothing)
    candidates = [i for i in ifaces if not _is_ap_iface(i)]
    if candidates:
        return candidates[0]

    return None


def ping_through_iface(iface: str, target: str, count: int = 1, timeout_s: int = 3) -> bool:
    if not shutil.which("ping"):
        print("✗ `ping` not found in PATH")
        return False

    # -I <iface> forces the outgoing interface
    # -c <count> send a small number of probes
    # -W <seconds> per-reply timeout (Linux ping)
    cp = _run(["ping", "-I", iface, "-c", str(count), "-W", str(timeout_s), target])
    if cp.returncode == 0:
        return True

    print("Ping failed.")
    if cp.stdout.strip():
        print(cp.stdout.strip())
    if cp.stderr.strip():
        print(cp.stderr.strip())
    return False


def main() -> bool:
    print("Network Ping Test (USB Wi-Fi preferred)")
    print("=" * 50)

    if not shutil.which("iw"):
        print("! `iw` not installed; interface type detection will be limited.")
        print("  Install with: sudo apt-get install -y iw")
    if not shutil.which("ip"):
        print("! `ip` not installed; link/address inspection will be limited.")
        print("  Install with: sudo apt-get install -y iproute2")

    ifaces = discover_wifi_ifaces()
    if not ifaces:
        print("✗ No wireless interfaces found under /sys/class/net/*/wireless")
        return False

    print("Detected Wi-Fi interfaces:")
    for i in ifaces:
        extra = []
        if i.is_usb:
            extra.append("USB")
        if i.iw_type:
            extra.append(f"type={i.iw_type}")
        if _is_ap_iface(i):
            extra.append(f"AP (expected SSID: {AP_SSID_NAME})")
        dev = i.dev_path or "unknown-device-path"
        print(f"  - {i.name}: {dev} [{', '.join(extra) if extra else 'no-metadata'}]")

    chosen = pick_usb_wifi_iface(ifaces)
    if not chosen:
        print("✗ Could not pick a suitable Wi-Fi interface (need a non-AP interface).")
        return False

    # Warn if we didn't manage to select USB
    if not chosen.is_usb:
        print(f"! No USB Wi-Fi interface detected; falling back to {chosen.name}.")

    link_up = _ip_link_is_up(chosen.name)
    if link_up is False:
        print(f"✗ Interface {chosen.name} is not UP")
        return False

    addrs = _ipv4_addrs(chosen.name)
    if addrs:
        print(f"Interface {chosen.name} IPv4: {', '.join(addrs)}")
    else:
        print(f"! Interface {chosen.name} has no IPv4 address (may still work with IPv6-only, but ping target is IPv4).")

    print(f"\nPinging {PING_TARGET} via {chosen.name} ...")
    ok = ping_through_iface(chosen.name, PING_TARGET, count=1, timeout_s=3)
    if ok:
        print("✓ Network looks good (ping succeeded via selected interface).")
        return True

    print("✗ Network test failed (ping did not succeed via selected interface).")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





