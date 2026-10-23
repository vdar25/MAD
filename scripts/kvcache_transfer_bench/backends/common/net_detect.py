"""
Network defaults for clusters that do not pin them: the RDMA device and the
socket interface. run_node.sh resolves both once per node and exports them.

Usage: net_detect.py rdma-device [CANDIDATES] | socket-ifname
"""

import os
import re
import sys
from pathlib import Path

SYS_IB = Path("/sys/class/infiniband")
KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _natural(name: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def socket_ifname() -> str:
    """Interface carrying the default route."""
    for line in _read(Path("/proc/net/route")).splitlines()[1:]:
        fields = line.split()
        if len(fields) > 1 and fields[1] == "00000000":
            return fields[0]
    return ""


def _gpu0_pci_path() -> str:
    """sysfs PCI path of HIP device 0; KFD lists GPU nodes in HIP order."""
    try:
        nodes = sorted((p for p in KFD_NODES.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    except OSError:
        return ""
    for node in nodes:
        props = dict(line.split()[:2] for line in _read(node / "properties").splitlines() if len(line.split()) > 1)
        if int(props.get("simd_count", 0)) == 0:
            continue
        loc, domain = int(props.get("location_id", 0)), int(props.get("domain", 0))
        bdf = f"{domain:04x}:{loc >> 8:02x}:{(loc >> 3) & 0x1f:02x}.{loc & 7:x}"
        return os.path.realpath(f"/sys/bus/pci/devices/{bdf}")
    return ""


def _shared_depth(a: str, b: str) -> int:
    depth = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        depth += 1
    return depth


def rdma_device(candidates: str = "") -> str:
    """The active RDMA device sharing the most PCIe ancestors with GPU 0.

    Every backend allocates on GPU 0, and this is the NIC MoRI-IO would pick for it.
    ``candidates`` (comma-separated) narrows the choice, and a single name is returned
    as given. The device behind the default-route interface is the frontend NIC and is
    skipped, as are bonds while any other device qualifies.
    """
    names = [c.strip() for c in candidates.split(",") if c.strip()]
    if len(names) == 1:
        return names[0]
    try:
        devices = sorted((p.name for p in SYS_IB.iterdir()), key=_natural)
    except OSError:
        return ""
    if names:
        devices = [d for d in devices if d in names]
    frontend = socket_ifname()

    def usable(dev: str) -> bool:
        if not any(_read(p).endswith("ACTIVE") for p in (SYS_IB / dev / "ports").glob("*/state")):
            return False
        try:
            return frontend not in os.listdir(SYS_IB / dev / "device" / "net")
        except OSError:
            return True

    found = [d for d in devices if usable(d)]
    found = [d for d in found if "bond" not in d] or found
    if not found:
        return ""
    gpu = _gpu0_pci_path()
    if not gpu:
        return found[0]
    return max(found, key=lambda d: _shared_depth(gpu, os.path.realpath(SYS_IB / d / "device")))


if __name__ == "__main__":
    if sys.argv[1:2] == ["rdma-device"]:
        print(rdma_device(sys.argv[2] if len(sys.argv) > 2 else ""))
    elif sys.argv[1:2] == ["socket-ifname"]:
        print(socket_ifname())
    else:
        sys.exit(__doc__.strip().splitlines()[-1])
