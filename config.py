"""
config.py — Shared configuration, lazy imports, and CRC tables.
All hardware modules import from here.
"""
import os
import json
import math

# ── Lazy-import globals ───────────────────────────────────────────
np       = None
Rotation = None
cKDTree  = None
rs       = None
SMBus    = None
cv2      = None

def _import_numpy():
    global np
    if np is None:
        import numpy as _np
        np = _np

def _import_scipy():
    global Rotation, cKDTree
    if Rotation is None:
        from scipy.spatial.transform import Rotation as _R
        from scipy.spatial import cKDTree as _T
        Rotation = _R
        cKDTree  = _T

def _import_realsense():
    global rs
    if rs is None:
        import pyrealsense2 as _rs
        rs = _rs

def _import_smbus():
    global SMBus
    if SMBus is None:
        from smbus2 import SMBus as _S
        SMBus = _S

def _import_cv2():
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2


# ── Default robot configuration ───────────────────────────────────
DEFAULT_CONFIG = {
    "LIDAR_OFFSET": 270,
    "LEFT_MOTOR_PORT":  "/dev/ttyACM1",
    "RIGHT_MOTOR_PORT": "/dev/ttyACM0",
    "LEFT_MOTOR_ID":  1,
    "RIGHT_MOTOR_ID": 3,
    "LEFT_SIGN":  1,
    "RIGHT_SIGN": -1,
    "TRIM": 0,

    # Camera mounting angles (degrees).
    #   CAMERA_YAW_DEG  : +N = camera rotated N° LEFT  of robot forward
    #   CAMERA_PITCH_DEG: +N = camera tilted  N° UPWARD from horizontal
    #   CAMERA_ROLL_DEG : +N = camera rolled  N° clockwise (viewed from behind)
    "CAMERA_YAW_DEG":   0.0,
    "CAMERA_PITCH_DEG": 0.0,
    "CAMERA_ROLL_DEG":  0.0,

    # Camera body position on robot (metres from base centre)
    "CAMERA_X": 0.10,
    "CAMERA_Y": 0.00,
    "CAMERA_Z": 0.15,

    # LiDAR body position
    "LIDAR_X": 0.00,
    "LIDAR_Y": 0.00,
    "LIDAR_Z": 0.20,
}


def load_config(path="robot_config.json"):
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
            print(f"[Config] Loaded {path}")
        except Exception as e:
            print(f"[Config] Error loading {path}: {e}")
    else:
        print(f"[Config] Using defaults (no {path})")
    return cfg


def save_config_keys(updates: dict, path="robot_config.json"):
    """Merge `updates` into existing JSON file (creates it if missing)."""
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(updates)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ── CRC helpers ───────────────────────────────────────────────────
CRC8_MAXIM_TABLE = [0] * 256
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x8C if _c & 1 else _c >> 1
    CRC8_MAXIM_TABLE[_i] = _c


def crc8_maxim(data: bytes) -> int:
    c = 0
    for b in data:
        c = CRC8_MAXIM_TABLE[c ^ b]
    return c


CRC_TABLE_LD06 = [
    0x00,0x4D,0x9A,0xD7,0x79,0x34,0xE3,0xAE,0xF2,0xBF,0x68,0x25,0x8B,0xC6,0x11,0x5C,
    0xA9,0xE4,0x33,0x7E,0xD0,0x9D,0x4A,0x07,0x5B,0x16,0xC1,0x8C,0x22,0x6F,0xB8,0xF5,
    0x1F,0x52,0x85,0xC8,0x66,0x2B,0xFC,0xB1,0xED,0xA0,0x77,0x3A,0x94,0xD9,0x0E,0x43,
    0xB6,0xFB,0x2C,0x61,0xCF,0x82,0x55,0x18,0x44,0x09,0xDE,0x93,0x3D,0x70,0xA7,0xEA,
    0x3E,0x73,0xA4,0xE9,0x47,0x0A,0xDD,0x90,0xCC,0x81,0x56,0x1B,0xB5,0xF8,0x2F,0x62,
    0x97,0xDA,0x0D,0x40,0xEE,0xA3,0x74,0x39,0x65,0x28,0xFF,0xB2,0x1C,0x51,0x86,0xCB,
    0x21,0x6C,0xBB,0xF6,0x58,0x15,0xC2,0x8F,0xD3,0x9E,0x49,0x04,0xAA,0xE7,0x30,0x7D,
    0x88,0xC5,0x12,0x5F,0xF1,0xBC,0x6B,0x26,0x7A,0x37,0xE0,0xAD,0x03,0x4E,0x99,0xD4,
    0x7C,0x31,0xE6,0xAB,0x05,0x48,0x9F,0xD2,0x8E,0xC3,0x14,0x59,0xF7,0xBA,0x6D,0x20,
    0xD5,0x98,0x4F,0x02,0xAC,0xE1,0x36,0x7B,0x27,0x6A,0xBD,0xF0,0x5E,0x13,0xC4,0x89,
    0x63,0x2E,0xF9,0xB4,0x1A,0x57,0x80,0xCD,0x91,0xDC,0x0B,0x46,0xE8,0xA5,0x72,0x3F,
    0xCA,0x87,0x50,0x1D,0xB3,0xFE,0x29,0x64,0x38,0x75,0xA2,0xEF,0x41,0x0C,0xDB,0x96,
    0x42,0x0F,0xD8,0x95,0x3B,0x76,0xA1,0xEC,0xB0,0xFD,0x2A,0x67,0xC9,0x84,0x53,0x1E,
    0xEB,0xA6,0x71,0x3C,0x92,0xDF,0x08,0x45,0x19,0x54,0x83,0xCE,0x60,0x2D,0xFA,0xB7,
    0x5D,0x10,0xC7,0x8A,0x24,0x69,0xBE,0xF3,0xAF,0xE2,0x35,0x78,0xD6,0x9B,0x4C,0x01,
    0xF4,0xB9,0x6E,0x23,0x8D,0xC0,0x17,0x5A,0x06,0x4B,0x9C,0xD1,0x7F,0x32,0xE5,0xA8,
]


# ── Sensor-to-base transform builders ────────────────────────────
def build_camera_transform(cfg):
    """
    Build T_base_cam (4×4): RealSense optical frame → robot base frame.

    Step 1 — Fixed optical-to-body remap (always required):
        RealSense optical: X=right, Y=down, Z=forward(depth)
        Robot base (ROS):  X=forward, Y=left, Z=up
        base_X =  cam_Z,  base_Y = -cam_X,  base_Z = -cam_Y

    Step 2 — User mounting offset (intrinsic ZYX Euler):
        yaw = Rz(+N → camera faces left), pitch = Ry, roll = Rx

    Step 3 — Translation (camera body position on robot).
    """
    _import_numpy()

    R_opt = np.array([
        [ 0.,  0.,  1.],   # base_X = cam_Z  (forward)
        [-1.,  0.,  0.],   # base_Y = -cam_X (left)
        [ 0., -1.,  0.],   # base_Z = -cam_Y (up)
    ], dtype=np.float64)

    yaw   = math.radians(cfg.get("CAMERA_YAW_DEG",   0.0))
    pitch = math.radians(cfg.get("CAMERA_PITCH_DEG", 0.0))
    roll  = math.radians(cfg.get("CAMERA_ROLL_DEG",  0.0))

    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)

    Rz = np.array([[cy,-sy,0.],[sy,cy,0.],[0.,0.,1.]], dtype=np.float64)
    Ry = np.array([[cp,0.,sp],[0.,1.,0.],[-sp,0.,cp]], dtype=np.float64)
    Rx = np.array([[1.,0.,0.],[0.,cr,-sr],[0.,sr,cr]], dtype=np.float64)

    R_total = Rz @ Ry @ Rx @ R_opt

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_total
    T[0, 3] = cfg.get("CAMERA_X", 0.10)
    T[1, 3] = cfg.get("CAMERA_Y", 0.00)
    T[2, 3] = cfg.get("CAMERA_Z", 0.15)
    return T


def build_lidar_transform(cfg):
    """T_base_lid: LiDAR sensor → robot base frame (rotation = identity)."""
    _import_numpy()
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = cfg.get("LIDAR_X", 0.00)
    T[1, 3] = cfg.get("LIDAR_Y", 0.00)
    T[2, 3] = cfg.get("LIDAR_Z", 0.20)
    return T