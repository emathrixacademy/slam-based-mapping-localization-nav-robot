"""
motors.py — Motor driver: Motor, DualMotors, and serial port auto-detection.
"""
import serial
import struct
import time
import glob

from config import crc8_maxim

MODE_VELOCITY = 0x02


class Motor:
    """Single brushless motor controller over UART."""

    def __init__(self, port: str, motor_id: int, name: str = "motor"):
        self.port      = port
        self.motor_id  = motor_id
        self.name      = name
        self.ser       = None
        self.connected = False
        try:
            self.ser = serial.Serial(port, 115200, timeout=0.2)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
            self.connected = True
            print(f"[{name}] Connected on {port} (ID {motor_id})")
        except Exception as e:
            print(f"[{name}] Failed on {port}: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _build(self, data9: list) -> bytes:
        raw = bytes(data9)
        return raw + bytes([crc8_maxim(raw)])

    def set_mode_velocity(self):
        if not self.connected: return
        pkt = bytes([self.motor_id, 0xA0, 0, 0, 0, 0, 0, 0, 0, MODE_VELOCITY])
        self.ser.write(pkt)
        self.ser.flush()
        time.sleep(0.05)
        self.ser.reset_input_buffer()

    def set_velocity(self, rpm: int):
        if not self.connected: return
        rpm = max(-330, min(330, int(rpm)))
        val = struct.pack(">h", rpm)
        pkt = self._build([self.motor_id, 0x64, val[0], val[1], 0, 0, 0, 0, 0])
        self.ser.write(pkt)
        self.ser.flush()
        self.ser.read(10)

    def stop(self):
        self.set_velocity(0)

    def brake(self):
        if not self.connected: return
        pkt = self._build([self.motor_id, 0x64, 0, 0, 0, 0, 0, 0xFF, 0])
        self.ser.write(pkt)
        self.ser.flush()
        self.ser.read(10)

    def get_info(self):
        if not self.connected: return None
        pkt = self._build([self.motor_id, 0x74, 0, 0, 0, 0, 0, 0, 0])
        self.ser.write(pkt)
        self.ser.flush()
        resp = self.ser.read(10)
        if len(resp) < 10: return None
        return {"id": resp[0], "velocity": struct.unpack(">h", resp[4:6])[0]}


class DualMotors:
    """Paired left/right motors with direction signs and trim."""

    def __init__(self, left_port, right_port, left_id, right_id,
                 left_sign=1, right_sign=-1, trim=0):
        self.left_sign  = left_sign
        self.right_sign = right_sign
        self.trim       = trim
        self.left  = Motor(left_port,  left_id,  "LEFT")
        self.right = Motor(right_port, right_id, "RIGHT")
        if self.left.connected:  self.left.set_mode_velocity()
        if self.right.connected: self.right.set_mode_velocity()

    def drive(self, left_rpm: int, right_rpm: int):
        al = int(left_rpm  * self.left_sign)  - self.trim
        ar = int(right_rpm * self.right_sign) + self.trim
        if self.left.connected:  self.left.set_velocity(al)
        if self.right.connected: self.right.set_velocity(ar)

    def stop(self):
        self.drive(0, 0)

    def brake(self):
        if self.left.connected:  self.left.brake()
        if self.right.connected: self.right.brake()

    def close(self):
        self.left.close()
        self.right.close()


def probe_motor_ports(left_id: int, right_id: int):
    """
    Scan /dev/ttyACM* and return (left_port, right_port) by sending a
    status-query packet and checking the echoed motor ID byte.
    """
    acm = sorted(glob.glob("/dev/ttyACM*"))
    port_to_id = {}
    for probe_port in acm:
        for test_id in [left_id, right_id]:
            try:
                ser = serial.Serial(probe_port, 115200, timeout=0.2)
                time.sleep(0.1)
                ser.reset_input_buffer()
                data9 = [test_id, 0x74, 0, 0, 0, 0, 0, 0, 0]
                raw   = bytes(data9)
                ser.write(raw + bytes([crc8_maxim(raw)]))
                ser.flush()
                resp = ser.read(10)
                ser.close()
                if len(resp) >= 10 and resp[0] == test_id:
                    port_to_id[probe_port] = test_id
                    break
            except Exception:
                try: ser.close()
                except Exception: pass

    left_port = right_port = None
    for p, mid in port_to_id.items():
        if mid == left_id:    left_port  = p
        elif mid == right_id: right_port = p
    return left_port, right_port