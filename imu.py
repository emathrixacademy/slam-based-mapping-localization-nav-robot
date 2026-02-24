"""
imu.py — ICM-20948 9-DoF IMU driver.
Provides roll / pitch / yaw via complementary filter (mag-assisted when available).
"""
import math
import struct
import time

from config import _import_numpy, _import_smbus

I2C_BUS  = 7
ICM_ADDR = 0x68
MAG_ADDR = 0x0C

G_TO_MS2    = 9.80665
DPS_TO_RADS = math.pi / 180.0


class ICM20948:
    ACCEL_SCALE = {0: 16384.0, 1: 8192.0, 2: 4096.0, 3: 2048.0}
    GYRO_SCALE  = {0: 131.0,   1: 65.5,   2: 32.8,   3: 16.4}

    def __init__(self, bus=I2C_BUS, addr=ICM_ADDR):
        _import_smbus()
        from config import SMBus
        self.bus    = SMBus(bus)
        self.addr   = addr
        self.afs    = 0
        self.gfs    = 1
        self.mag_ok = False
        self._bank  = -1

        self.roll = self.pitch = self.yaw = 0.0
        self._last_t = time.time()
        self._first  = True
        self._alpha  = 0.98

    # ── Register I/O ─────────────────────────────────────────────
    def _set_bank(self, b):
        if self._bank != b:
            self.bus.write_byte_data(self.addr, 0x7F, (b & 3) << 4)
            self._bank = b

    def _rd(self, reg, bank=0):
        self._set_bank(bank)
        return self.bus.read_byte_data(self.addr, reg)

    def _wr(self, reg, val, bank=0):
        self._set_bank(bank)
        self.bus.write_byte_data(self.addr, reg, val)

    def _rd_blk(self, reg, n, bank=0):
        self._set_bank(bank)
        return self.bus.read_i2c_block_data(self.addr, reg, n)

    # ── Initialisation ───────────────────────────────────────────
    def init(self):
        wai = self._rd(0x00)
        if wai != 0xEA:
            raise RuntimeError(f"ICM-20948 not found (WAI=0x{wai:02X})")
        self._wr(0x06, 0x80); time.sleep(0.1)   # soft reset
        self._bank = -1
        self._wr(0x06, 0x01); time.sleep(0.05)  # auto-clock
        self._wr(0x07, 0x00)                     # enable accel + gyro
        self._wr(0x14, (self.afs << 1) | 1, bank=2)
        self._wr(0x01, (self.gfs << 1) | 1, bank=2)
        self._wr(0x0F, 0x02); time.sleep(0.01)
        try:
            mid = self.bus.read_byte_data(MAG_ADDR, 0x01)
            if mid == 0x09:
                self.bus.write_byte_data(MAG_ADDR, 0x31, 0x08)
                time.sleep(0.01)
                self.mag_ok = True
        except Exception:
            pass
        self._set_bank(0)

    # ── Reading ──────────────────────────────────────────────────
    def read_all(self) -> dict:
        # Accelerometer
        raw_a = self._rd_blk(0x2D, 6)
        ax, ay, az = struct.unpack('>hhh', bytes(raw_a))
        s = self.ACCEL_SCALE[self.afs]
        ax, ay, az = ax/s*G_TO_MS2, ay/s*G_TO_MS2, az/s*G_TO_MS2

        # Gyroscope
        raw_g = self._rd_blk(0x33, 6)
        gx, gy, gz = struct.unpack('>hhh', bytes(raw_g))
        s = self.GYRO_SCALE[self.gfs]
        gx, gy, gz = gx/s*DPS_TO_RADS, gy/s*DPS_TO_RADS, gz/s*DPS_TO_RADS

        # Temperature
        raw_t = self._rd_blk(0x39, 2)
        temp = (struct.unpack('>h', bytes(raw_t))[0] - 21.0) / 333.87 + 21.0

        # Magnetometer
        mx = my = mz = 0.0
        if self.mag_ok:
            try:
                st = self.bus.read_byte_data(MAG_ADDR, 0x10)
                if st & 1:
                    raw_m = self.bus.read_i2c_block_data(MAG_ADDR, 0x11, 8)
                    mx, my, mz = struct.unpack('<hhh', bytes(raw_m[:6]))
                    mx, my, mz = mx*0.15, my*0.15, mz*0.15
            except Exception:
                pass

        # Complementary filter
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now

        if self._first or dt > 1.0:
            self.roll  = math.atan2(ay, az)
            self.pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
            self.yaw   = 0.0
            self._first = False
        else:
            a = self._alpha
            self.roll  = a*(self.roll  + gx*dt) + (1-a)*math.atan2(ay, az)
            self.pitch = a*(self.pitch + gy*dt) + (1-a)*math.atan2(-ax, math.sqrt(ay*ay+az*az))
            self.yaw  += gz * dt
            if self.mag_ok and (mx != 0 or my != 0):
                cr, sr = math.cos(self.roll),  math.sin(self.roll)
                cp, sp = math.cos(self.pitch), math.sin(self.pitch)
                mag_yaw = math.atan2(
                    -(my*cr - mz*sr),
                    mx*cp + my*sr*sp + mz*cr*sp
                )
                self.yaw = a*self.yaw + (1-a)*mag_yaw

        return {
            "accel": (ax, ay, az),
            "gyro":  (gx, gy, gz),
            "mag":   (mx, my, mz),
            "temp":  temp,
            "roll":  self.roll,
            "pitch": self.pitch,
            "yaw":   self.yaw,
        }

    def get_orientation_matrix(self):
        """Return 4×4 homogeneous rotation matrix from current r/p/y."""
        _import_numpy()
        from config import np as _np
        cr, sr = math.cos(self.roll),  math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw),   math.sin(self.yaw)
        R = _np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr            ],
        ], dtype=_np.float64)
        T = _np.eye(4, dtype=_np.float64)
        T[:3, :3] = R
        return T

    def close(self):
        self.bus.close()