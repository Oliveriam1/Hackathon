"""Příjem telemetrie míření z dronu (UDP JSON) – spusťte na notebooku.

  python3 tools/receive_telemetry.py --port 5005
  python3 tools/receive_telemetry.py --port 5005 --log let.jsonl   # zároveň uložit

Na dronu:  python3 main.py ... --send <IP notebooku>:5005
IP notebooku zjistíte příkazem ipconfig (Windows) / ip addr (Linux).
Windows Firewall se může zeptat, zda povolit Python v síti – povolte.
"""
import argparse
import json
import socket
import time
from pathlib import Path


def describe(message):
    if message.get('type') == 'result':
        angles = message['angles_deg']
        text = (f"VÝSLEDEK  úhel right {angles['right']:+.2f}°, forward {angles['forward']:+.2f}° "
                f"({message['samples']} snímků, rozptyl {message['spread_deg']:.2f}°)")
        target = message.get('target')
        if target:
            text += (f"\n          TEČKA {target['latitude_deg']:.8f}, {target['longitude_deg']:.8f} "
                     f"({target['distance_m']:.2f} m od dronu, ±{target['error_m']*100:.0f} cm)")
        return text
    camera = message.get('camera_angles_deg') or {}
    text = f"{message.get('state', '?'):10s} kamera R={camera.get('right', 0):+6.2f} F={camera.get('forward', 0):+6.2f}"
    target = message.get('target_angles_deg')
    if target:
        text += f" | úhel na tečku R={target['right']:+6.2f} F={target['forward']:+6.2f}"
    if message.get('pixel_error'):
        text += f" | odchylka {message['pixel_error'][0]:+.0f},{message['pixel_error'][1]:+.0f} px"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--log', type=Path, help='Ukládat všechny zprávy jako JSON Lines.')
    args = parser.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', args.port))
    sock.settimeout(1.)
    print(f'Poslouchám UDP port {args.port} ... (Ctrl+C ukončí)')
    log = args.log.open('a', encoding='utf-8') if args.log else None
    last_seen = None
    last_seq = None
    lost = 0
    try:
        while True:
            try:
                data, sender = sock.recvfrom(65535)
            except socket.timeout:
                if last_seen is not None and time.monotonic()-last_seen > 3:
                    print(f'... žádná data {time.monotonic()-last_seen:.0f} s')
                continue
            try:
                message = json.loads(data.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                print(f'Neplatná zpráva od {sender[0]}')
                continue
            seq = message.get('seq')
            if last_seq is not None and isinstance(seq, int) and seq > last_seq+1:
                lost += seq-last_seq-1
            last_seq, last_seen = seq, time.monotonic()
            print(f"[{sender[0]}] {describe(message)}" + (f'  (ztraceno {lost})' if lost else ''), flush=True)
            if log is not None:
                log.write(json.dumps(message, ensure_ascii=False)+'\n')
                log.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if log is not None:
            log.close()
        sock.close()


if __name__ == '__main__':
    main()
