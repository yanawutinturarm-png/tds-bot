"""Records how the player plays when the bot hands a wave over (e.g. the final wave 35).

While a recording runs, the player's mouse clicks and game keys inside Roblox are written to
recordings/<date-time>-wave<N>/events.jsonl with the cash, wave and tower count at that moment and
the click's position on the map. A picture of the screen is saved at every click (the click marked)
and every few seconds, plus the bot's towers at the handover. Nothing is sent to the game.

Only keys the game uses are recorded (see GAME_KEYS), and only while Roblox is the focused window,
so typing anywhere else is never logged.
"""
from __future__ import annotations

import ctypes
import json
import threading
import time
from collections import Counter
from ctypes import wintypes
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
GAME_KEYS = "1234567890QEFBXRCVZGTH"  # tower keys, upgrade, abilities, sell, cancel, ...
MOUSE = {0x01: "LMB", 0x02: "RMB", 0x04: "MMB"}

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short
_user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]


def _key_down(vk):
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def _cursor():
    p = wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


class HumanRecorder(threading.Thread):
    """Watches the player's input all the time (so the bot knows someone took over) and writes it
    to disk while a recording is on."""

    def __init__(self, cfg, latest_frame, get_region, focused, to_map, reading,
                 key_down=_key_down, cursor=_cursor):
        super().__init__(name="recorder", daemon=True)
        c = cfg.get("handover", {})
        self.save = c.get("record", True)
        self.frame_every = c.get("frame_every_s", 3.0)
        self.keys = {**MOUSE, **{ord(k): k for k in c.get("keys", GAME_KEYS).upper()}}
        self.latest_frame, self.get_region, self.focused = latest_frame, get_region, focused
        self.to_map, self.reading = to_map, reading
        self.key_down, self.cursor = key_down, cursor
        self.last_input = 0.0      # monotonic time of the player's last click or game key (Roblox focused)
        self.dir = None            # folder of the recording in progress
        self.counts = Counter()
        self._lock = threading.RLock()  # start_recording stops the previous one under the same lock
        self._halt = threading.Event()
        self._events = None
        self._rec_t0 = self._next_frame = 0.0
        self._n = 0
        self._start_cash = -1

    # ---- the bot calls these
    def start_recording(self, wave, towers):
        """Begin a recording: the bot just handed wave `wave` over. `towers`: the bot's towers then."""
        with self._lock:
            self.stop_recording("a new recording started")
            self.counts = Counter()
            self._rec_t0 = time.monotonic()
            self._next_frame = 0.0
            self._n = 0
            r = self.reading()
            self._start_cash = r.cash if r else -1
            if not self.save:
                self.dir = None
                return None
            self.dir = ROOT / "recordings" / f"{time.strftime('%Y%m%d-%H%M%S')}-wave{wave}"
            (self.dir / "frames").mkdir(parents=True, exist_ok=True)
            (self.dir / "towers.json").write_text(json.dumps(towers, indent=1), encoding="utf-8")
            self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
            self._write({"kind": "start", "wave": wave})
            self._snapshot("start")
            return self.dir

    def stop_recording(self, result=""):
        """End the recording in progress (match over, nobody took over, bot stopped)."""
        with self._lock:
            if self._events is None:
                return None
            self._snapshot("end")
            self._write({"kind": "end", "result": result})
            self._events.close()
            self._events = None
            r = self.reading()
            summary = {"result": result, "seconds": round(time.monotonic() - self._rec_t0, 1),
                       "cash_at_start": self._start_cash, "cash_at_end": r.cash if r else -1,
                       "presses": dict(self.counts.most_common()), "pictures": self._n}
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
            done, self.dir = self.dir, None
            return done

    @property
    def recording(self):
        return self._events is not None

    def halt(self):
        self._halt.set()
        with self._lock:
            self.stop_recording("the bot stopped")

    # ---- the thread
    def run(self):
        was = {}
        while not self._halt.is_set():
            focused = self.focused()
            for vk, name in self.keys.items():
                down = self.key_down(vk)
                if down != was.get(vk, False) and focused:
                    if down:
                        self.last_input = time.monotonic()
                    with self._lock:
                        if self._events is not None:
                            self._input(name, down)
                was[vk] = down
            with self._lock:
                if self._events is not None and time.monotonic() >= self._next_frame:
                    self._state()
            self._halt.wait(0.01)

    # ---- inside, with the lock held
    def _write(self, event):
        r = self.reading()
        event = {"t": round(time.time(), 3), "s": round(time.monotonic() - self._rec_t0, 2), **event}
        if r is not None:
            event.update(cash=r.cash, wave=r.wave, towers=r.towers)
        self._events.write(json.dumps(event) + "\n")
        self._events.flush()

    def _input(self, name, down):
        region = self.get_region()
        x, y = self.cursor()
        event = {"kind": "down" if down else "up", "key": name, "x": x, "y": y}
        if region:
            fx, fy = (x - region["left"]) / region["width"], (y - region["top"]) / region["height"]
            event.update(fx=round(fx, 4), fy=round(fy, 4))
            if down and name in ("LMB", "RMB"):
                mx, my = self.to_map(x, y, region)
                event["map"] = [round(mx, 4), round(my, 4)]  # where on the map, if the camera had moved
        if down:
            self.counts[name] += 1
            if name in ("LMB", "RMB"):
                event["picture"] = self._snapshot(f"{name.lower()}", (event.get("fx"), event.get("fy")))
        self._write(event)

    def _state(self):
        self._next_frame = time.monotonic() + self.frame_every
        self._write({"kind": "state", "picture": self._snapshot("every")})

    def _snapshot(self, why, mark=None):
        frame = self.latest_frame()
        if frame is None or self.dir is None:
            return None
        img = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])), interpolation=cv2.INTER_AREA)
        if mark and mark[0] is not None:
            c = (int(mark[0] * img.shape[1]), int(mark[1] * img.shape[0]))
            cv2.circle(img, c, 14, (0, 0, 255), 2)
            cv2.circle(img, c, 2, (0, 0, 255), -1)
        self._n += 1
        name = f"{self._n:04d}-{why}.jpg"
        cv2.imwrite(str(self.dir / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return name
