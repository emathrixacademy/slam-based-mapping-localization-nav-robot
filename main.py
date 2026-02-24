#!/usr/bin/env python3
"""
main.py — SLAM Avoidance Robot entry point.

Usage:
    python3 main.py                  # navigate + map
    python3 main.py --map-only       # map only, motors off
    python3 main.py check            # hardware connection check
    python3 main.py --port 8081      # custom dashboard port

Project layout:
    main.py          ← this file (CLI + startup)
    config.py        ← shared config, CRC tables, transform builders
    motors.py        ← Motor, DualMotors, probe_motor_ports
    lidar.py         ← LD06 driver + point-cloud utilities
    imu.py           ← ICM-20948 driver
    camera.py        ← RealSense D4xx driver
    robot.py         ← SLAMAvoidanceRobot + VFHNavigator
    server.py        ← HTTP dashboard server
    dashboard.html   ← frontend (Three.js 3-D viewer, 2-D map, radar)
"""

import os
import sys
import glob
import socket

from config import load_config


# ═══════════════════════════════════════════════════════════════════
#  Hardware connection check  (python3 main.py check)
# ═══════════════════════════════════════════════════════════════════
def run_check(config: dict):
    from motors import probe_motor_ports
    from config import _import_realsense

    print("=" * 50)
    print("  SLAM Robot — Hardware Check")
    print("=" * 50)

    acm = sorted(glob.glob("/dev/ttyACM*"))
    print(f"Serial ports:  {acm or 'none'}")

    lidar_port = "/dev/ttyTHS1"
    print(f"LiDAR port:    {lidar_port}  "
          f"{'[OK]' if os.path.exists(lidar_port) else '[NOT FOUND]'}")

    i2c7 = "/dev/i2c-7"
    print(f"I2C-7 (IMU):   {i2c7}  "
          f"{'[OK]' if os.path.exists(i2c7) else '[NOT FOUND]'}")

    lp, rp = probe_motor_ports(config["LEFT_MOTOR_ID"], config["RIGHT_MOTOR_ID"])
    print(f"Left  motor (ID {config['LEFT_MOTOR_ID']}):  {lp or 'NOT FOUND'}")
    print(f"Right motor (ID {config['RIGHT_MOTOR_ID']}):  {rp or 'NOT FOUND'}")

    try:
        _import_realsense()
        from config import rs
        devs = rs.context().query_devices()
        print(f"RealSense:     {len(devs)} device(s) connected")
    except Exception as e:
        print(f"RealSense:     not available ({e})")

    print()
    print("Camera transform (yaw/pitch/roll=0°):")
    from config import build_camera_transform
    T = build_camera_transform(config)
    print("  Rotation matrix:")
    for row in T[:3, :3]:
        print(f"    [{row[0]:+.3f}  {row[1]:+.3f}  {row[2]:+.3f}]")
    print(f"  Translation: {T[:3,3]}")
    fwd = T[:3, :3] @ [0, 0, 1]   # cam_Z (depth) should map to base_X (+forward)
    print(f"  cam_Z → base: [{fwd[0]:+.2f}, {fwd[1]:+.2f}, {fwd[2]:+.2f}]"
          f"  {'✓ forward' if abs(fwd[0]-1)<0.01 else '✗ CHECK MOUNTING ANGLES'}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    config   = load_config("robot_config.json")
    args     = sys.argv[1:]
    map_only = "--map-only" in args

    # Parse --port N
    dash_port = 8080
    if "--port" in args:
        try:
            dash_port = int(args[args.index("--port") + 1])
        except (IndexError, ValueError):
            print("[main] Invalid --port value, using 8080")

    # Hardware check subcommand
    positional = [a for a in args if not a.startswith("-")]
    if positional and positional[0] == "check":
        run_check(config)
        return

    # Resolve local IP for the startup banner
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print("=" * 60)
    print("  SLAM Avoidance Robot")
    print(f"  Mode:      {'MAP ONLY' if map_only else 'NAVIGATE + MAP'}")
    print(f"  LiDAR:     offset={config['LIDAR_OFFSET']}°")
    print(f"  Dashboard: http://localhost:{dash_port}")
    print(f"  Network:   http://{local_ip}:{dash_port}")
    print("=" * 60)

    from robot  import SLAMAvoidanceRobot
    from server import start_dashboard

    robot = SLAMAvoidanceRobot(config, map_only=map_only)
    start_dashboard(robot, dash_port)

    try:
        robot.run()
    finally:
        robot.close()


if __name__ == "__main__":
    main()