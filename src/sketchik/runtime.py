"""Local configuration and explicit GPU-to-EGL binding, before MuJoCo import."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import uuid


def configure(root: Path, gpu_uuid: str, variant="pro") -> dict:
    if "mujoco" in sys.modules:
        raise RuntimeError("Configure renderer before importing MuJoCo / primitives")
    root = root.resolve()
    if variant not in ("pro", "standard"):
        raise ValueError("Unknown LIBERO variant")
    for name, folder in {"LIBERO_CONFIG_PATH": "libero_"+variant, "NUMBA_CACHE_DIR": "numba",
                         "MPLCONFIGDIR": "matplotlib"}.items():
        path = root / ".runtime" / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    selected = next((x.split(",") for x in rows.splitlines() if x.split(",")[0].strip() == gpu_uuid), None)
    if selected is None:
        raise RuntimeError("Requested GPU UUID does not exist")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    occupied = {int(row.split(',')[1]) for row in apps.splitlines() if row.split(',')[0].strip() == gpu_uuid}
    if occupied or (not occupied and (int(selected[2]) > 512 or int(selected[3]) > 5)):
        raise RuntimeError(f"GPU is busy; refusing launch: {selected}")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # EGL's CUDA ordinal is the driver ordinal, not necessarily nvidia-smi's index.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    driver = C.CDLL("libcuda.so.1")
    def check(code):
        if code:
            raise RuntimeError(f"CUDA enumeration error {code}")
    check(driver.cuInit(0))
    count = C.c_int()
    check(driver.cuDeviceGetCount(C.byref(count)))
    ordinal = None
    for i in range(count.value):
        device, data = C.c_int(), (C.c_ubyte * 16)()
        check(driver.cuDeviceGet(C.byref(device), i))
        check(driver.cuDeviceGetUuid(C.byref(data), device))
        if "GPU-" + str(uuid.UUID(bytes=bytes(data))) == gpu_uuid:
            ordinal = i
    if ordinal is None:
        raise RuntimeError("Requested GPU absent from CUDA driver enumeration")
    from mujoco.egl import egl_ext
    from OpenGL import EGL
    from OpenGL.EGL.EXT.device_base import eglQueryDeviceStringEXT
    proc = EGL.eglGetProcAddress("eglQueryDeviceAttribEXT")
    if not proc:
        raise RuntimeError("Cannot verify EGL device mapping")
    query = C.cast(int(proc), C.CFUNCTYPE(C.c_uint, C.c_void_p, C.c_int, C.POINTER(C.c_ssize_t)))
    matches = []
    for index, dev in enumerate(egl_ext.eglQueryDevicesEXT()):
        extensions = (eglQueryDeviceStringEXT(dev, 0x3055) or b"").decode()
        value = C.c_ssize_t(-1)
        if "EGL_NV_device_cuda" in extensions and query(C.cast(dev, C.c_void_p), 0x323A, C.byref(value)):
            if value.value == ordinal:
                matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"Cannot uniquely map GPU to EGL: {matches}")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(matches[0])
    # No ML inference is needed; renderer uses the separately selected EGL device.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return {"uuid": gpu_uuid, "name": selected[1].strip(), "driver_ordinal": ordinal,
            "egl_device": matches[0], "preflight": rows.strip()}
