#!/usr/bin/env python3
"""receive.py - run on your laptop to print the telemetry sent by tracker.py"""
import json
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("", 5005))
print("listening on UDP 5005 ...")
while True:
    d = json.loads(s.recv(1024))
    if d["found"]:
        print(f"right {d['right_deg']:+6.2f}°  forward {d['fwd_deg']:+6.2f}°  "
              f"total {d['total_deg']:5.2f}°  ({d['fps']} fps)")
    else:
        print(f"no target  ({d['fps']} fps)")
