"""Windows side of the bot: Roblox window tracking, global failsafe hotkeys,
and the executor that turns decisions into DirectInput mouse/key events."""
from __future__ import annotations

import ctypes
import json
import math
import threading
import time
from collections import Counter, deque
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydirectinput as pdi

from placement import Prober, sell_button_visible

# Own user32 handle so these argtypes don't leak into pydirectinput's windll.user32.
user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.GetForegroundWindow.restype = wintypes.HWND
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
kernel32 = ctypes.WinDLL("kernel32")

PLACE_CONFIRM_S = 3.0   # how long the tower counter gets to show a new tower before the placement counts as failed
MAX_SPOT_FAILURES = 2   # a spot that fails this often is skipped (blocked, on the path, too close to another tower)
MAX_WAVE = 60           # no TDS mode has more waves: a bigger wave number is a misread
MAX_SPOTS = 80          # cap on calibrated plus automatically added spots
PANEL_SETTLE_S = 0.8    # the upgrade panel slides in; only frames this long after a click show it

KNOWN_ACTIONS ={f"PLACE_TOWER_{n}" for n in range(1, 6)} | {"UPGRADE_SELECTED", "SKIP_WAVE", "SELL", "WAIT"}

VK_NAMES = {"ESC": 0x1B, "ESCAPE": 0x1B, "PAUSE": 0x13, "END": 0x23, "HOME": 0x24,
            "INSERT": 0x2D, "DELETE": 0x2E, "BACKSPACE": 0x08, "TAB": 0x09,
            **{f"F{i}": 0x6F + i for i in range(1, 13)}}


def enable_dpi_awareness():
    """Use physical pixels everywhere so mss, Win32 and pydirectinput agree under display scaling."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def window_title(hwnd):
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def find_window(title):
    return user32.FindWindowW(None, title)


def client_region(hwnd):
    """Screen rectangle of a window's client area (the game, without title bar/borders)."""
    if not hwnd or not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)) or not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width < 200 or height < 200:
        return None
    return {"left": origin.x, "top": origin.y, "width": width, "height": height}


def focus_window(hwnd):
    """Bring a window to the front. Windows only lets the foreground app do that, so borrow its input queue."""
    if not hwnd:
        return False
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE, only when minimized (it would un-maximize otherwise)
    fg = user32.GetForegroundWindow()
    if fg == hwnd:
        return True
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    me = kernel32.GetCurrentThreadId()
    attached = bool(fg_thread and fg_thread != me and user32.AttachThreadInput(me, fg_thread, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(me, fg_thread, False)
    time.sleep(0.15)
    return user32.GetForegroundWindow() == hwnd


def ready_button_visible(frame, roi):
    """TDS's pre-game READY button is a saturated lime green that grass and the HUD never reach."""
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    crop = frame[int(y * h):int((y + rh) * h), int(x * w):int((x + rw) * w)]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 45) & (hsv[..., 0] <= 70) & (hsv[..., 1] >= 220) & (hsv[..., 2] >= 170)
    return green.mean() >= 0.5


ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


def keep_awake(on):
    """While the bot plays, ask Windows not to sleep or turn the screen off (the capture needs the
    screen). Only a request from this program: Windows drops it when the bot stops."""
    try:
        flags = ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | flags)
    except (AttributeError, OSError):
        pass


BOT_MUTEX = r"Local\TDSBotPlaying"
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateMutexW.restype = wintypes.HANDLE
_kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
_kernel32.OpenMutexW.restype = wintypes.HANDLE
_kernel32.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
_held_mutex = []


def hold_single_instance():
    """Claim the "a bot is playing" flag for this process (Windows drops it when the process ends).
    False if another bot already holds it: two bots on one game fight over the mouse and keys
    (it happened once, and every match was lost by wave 11)."""
    handle = _kernel32.CreateMutexW(None, False, BOT_MUTEX)
    if handle and ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        _kernel32.CloseHandle(handle)
        return False
    _held_mutex.append(handle)
    return True


def bot_running():
    """Is a bot already playing (holding the flag)?"""
    handle = _kernel32.OpenMutexW(0x00100000, False, BOT_MUTEX)  # SYNCHRONIZE
    if handle:
        _kernel32.CloseHandle(handle)
        return True
    return False


OWN_PRESSES = {}  # virtual-key code -> until when a press of it is the bot's own, not the user's


def end_screen_button(frame):
    """Where the green "Restart Match" button of TDS's end-of-match screen (YOU LOST, or a win) is, as
    fractions of the frame, or None. It sits under the big dark result panel; the green READY
    button and the health bar have no such panel above them."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] >= 40) & (hsv[..., 0] <= 80) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 140)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # fill in the white text
    dark = hsv[..., 2] < 50
    _, _, stats, _ = cv2.connectedComponentsWithStats(green)
    for x, y, bw, bh, area in stats[1:]:
        if not (0.09 * w <= bw <= 0.45 * w and 0.03 * h <= bh <= 0.12 * h and 2.5 <= bw / bh <= 8):
            continue
        if y < 0.45 * h or area < 0.4 * bw * bh:
            continue
        above = dark[max(0, y - h // 2):y, max(0, x - bw):min(w, x + bw)]
        if above.size and above.mean() >= 0.4:
            return (x + bw / 2) / w, (y + bh / 2) / h
    return None


def boss_bar_visible(frame):
    """Is a boss health bar up (the purple bar under the base health, e.g. "Molten Golem")? Still found
    when the boss is nearly dead and only a short piece of purple is left."""
    if frame is None:
        return False
    h, w = frame.shape[:2]
    roi = frame[int(0.13 * h):int(0.21 * h), int(0.3 * w):int(0.7 * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    purple = ((hsv[..., 0] >= 125) & (hsv[..., 0] <= 165) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 120)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(purple)
    return any(bw >= 0.012 * w and 0.006 * h <= bh <= 0.035 * h for _, _, bw, bh, _ in stats[1:])


def lost_title(frame, button):
    """On the result screen: is the title the red "YOU LOST" (not the gold-on-red "TRIUMPH!")?
    (`button` = the Restart Match / Play Again button.)"""
    h, w = frame.shape[:2]
    bx, by = int(button[0] * w), int(button[1] * h)
    band = frame[max(0, by - int(0.5 * h)):max(0, by - int(0.3 * h)), max(0, bx - int(0.35 * w)):min(w, bx + int(0.15 * w))]
    if band.size == 0:
        return False
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] <= 12) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 150)
    gold = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 180)
    # A win is gold "TRIUMPH!" on a wide red banner (red 0.17, gold 0.04 of the band); a loss is red
    # "YOU LOST" on black (red 0.03-0.07, gold at most 0.02). Both have red, so the red alone can't tell.
    if float(red.mean()) >= 0.12 and float(gold.mean()) >= 0.03:
        return False
    return float(red.mean()) >= 0.02


class CameraTracker:
    """How the game's view has changed since the match began: a boss skill shakes the camera, and the
    final Warlord slowly turns it while the light goes to sunset. Map clicks follow the change. Corner
    points (ORB) of a small grey copy are matched against the first in-match frame, the HUD and the
    upgrade panel left out, and a turn + zoom + shift is fitted (RANSAC). A fit with too few matching
    points is ignored, never acted on."""

    W, H = 960, 540

    def __init__(self):
        self.orb = cv2.ORB_create(2500, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        m = np.full((self.H, self.W), 255, np.uint8)
        m[:int(0.15 * self.H)] = 0                                   # wave, base health, boss bar
        m[int(0.86 * self.H):] = 0                                   # hotbar, cash, items
        m[:int(0.25 * self.H), int(0.78 * self.W):] = 0              # players list, settings
        m[int(0.42 * self.H):int(0.58 * self.H), :int(0.11 * self.W)] = 0   # ability buttons (F, B)
        m[int(0.58 * self.H):int(0.89 * self.H), int(0.33 * self.W):int(0.67 * self.W)] = 0   # upgrade panel
        self.mask = m
        self.reset()

    def reset(self):
        self.ref, self.M, self.Minv = None, None, None   # M: reference pixels -> pixels now (None = unchanged)
        self.moving_until, self._lost_since, self._moving_since = 0.0, None, None

    @property
    def offset(self):
        """The shift of the middle of the view, as fractions (0, 0 = unchanged)."""
        x, y = self.to_screen(0.5, 0.5)
        return x - 0.5, y - 0.5

    def to_screen(self, fx, fy):
        if self.M is None:
            return fx, fy
        x, y = self.M @ (fx * self.W, fy * self.H, 1.0)
        return x / self.W, y / self.H

    def to_map(self, fx, fy):
        if self.Minv is None:
            return fx, fy
        x, y = self.Minv @ (fx * self.W, fy * self.H, 1.0)
        return x / self.W, y / self.H

    def _features(self, frame):
        g = cv2.cvtColor(cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(g, self.mask)

    def update(self, frame, in_match=True):
        if frame is None:
            return
        kps, desc = self._features(frame)
        if self.ref is None:
            if in_match and desc is not None and len(kps) >= 200:
                self.ref = (kps, desc)
            return
        now = time.monotonic()
        fit, pts = self._fit(kps, desc)
        if fit is None:  # unsure (a big popup, the result screen): ignore, and start over if it lasts
            self._lost_since = self._lost_since or now
            if now - self._lost_since > 30:
                self.reset()
            return
        self._lost_since = None
        corners = np.float32([[0, 0, 1], [self.W, 0, 1], [0, self.H, 1], [self.W, self.H, 1]]).T
        # How far the map itself moved since the last fit, measured on the matched map points (the far
        # screen corners exaggerate a turn). Over 6 px: a shake, clicks wait. But a steady turn (the
        # Warlord's sunset turns the view for over a minute) isn't a shake: after 3 s of moving the
        # clicks go on, following the fit, instead of waiting out the whole fight with the cash piling up.
        before = self.M if self.M is not None else np.float32([[1, 0, 0], [0, 1, 0]])
        moved = float(np.percentile(np.hypot(*(fit @ pts - before @ pts)), 90)) * 2
        if moved > 6:
            self._moving_since = self._moving_since or now
            if now - self._moving_since < 3.0:
                self.moving_until = now + 1.0
        else:
            self._moving_since = None
        still = float(np.abs(fit @ corners - corners[:2]).max()) * 2 <= 3
        self.M = None if still else fit
        self.Minv = None if still else cv2.invertAffineTransform(fit)

    def _fit(self, kps, desc):
        """(turn + zoom + shift from the reference, the matching reference points as 3 x N), or (None, None)."""
        if desc is None or len(kps) < 50:
            return None, None
        rk, rd = self.ref
        pairs = self.matcher.knnMatch(rd, desc, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 40:
            return None, None
        src = np.float32([rk[m.queryIdx].pt for m in good])
        dst = np.float32([kps[m.trainIdx].pt for m in good])
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
        if M is None or inliers is None:
            return None, None
        n = int(inliers.sum())
        angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        scale = math.hypot(M[0, 0], M[1, 0])
        if n < 40 or n < 0.35 * len(good) or abs(angle) > 45 or not 0.7 <= scale <= 1.4:
            return None, None
        pts = src[inliers.ravel() == 1]
        return M.astype(np.float32), np.vstack([pts.T, np.ones(len(pts), np.float32)])


def _beep():
    """Three short beeps: the player's turn."""
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1000, 180)
            time.sleep(0.08)
    except (ImportError, RuntimeError):
        pass


def vk_code(name):
    key = name.strip().upper()
    if key in VK_NAMES:
        return VK_NAMES[key]
    if len(key) == 1 and key.isascii() and key.isalnum():
        return ord(key)
    raise ValueError(f"unsupported hotkey {name!r}")


def validate_config(cfg):
    """Fail fast on config mistakes instead of mid-game."""
    problems = []
    actions = cfg.get("actions", [])
    unknown = [a for a in actions if a not in KNOWN_ACTIONS]
    if unknown:
        problems.append(f"unknown actions {unknown}; allowed: {sorted(KNOWN_ACTIONS)}")
    if "WAIT" not in actions:
        actions.append("WAIT")
    keys = cfg.get("keys", {})
    for a in actions:
        if a.startswith("PLACE_TOWER_") and f"tower_{a[-1]}" not in keys:
            problems.append(f"{a} needs keys.tower_{a[-1]}")
    for name, key in keys.items():
        if key not in pdi.KEYBOARD_MAPPING:
            problems.append(f"keys.{name} = {key!r} is not a key pydirectinput can press")
    hotkeys = {cfg["hotkeys"]["pause"].upper(), cfg["hotkeys"]["kill"].upper()}
    for name in hotkeys:
        try:
            vk_code(name)
        except ValueError as e:
            problems.append(str(e))
    n_spots = len(cfg.get("placement_spots", []))
    for slot, t in cfg.get("towers", {}).items():
        bad = [n for n in t.get("spots") or [] if not isinstance(n, int) or not 1 <= n <= n_spots]
        if bad:
            problems.append(f"towers.{slot}.spots {bad}: use spot numbers 1-{n_spots} as shown in the preview")
        for key in ("limit", "from_wave", "until_wave", "save_for", "upgrade_to"):
            if not isinstance(t.get(key, 0), int) or t.get(key, 0) < 0:
                problems.append(f"towers.{slot}.{key} must be a whole number")
    for i, step in enumerate(parse_plan(cfg), 1):
        if step["slot"] is None:
            problems.append(f"plan step {i}: unknown tower {step['tower']!r} (use a name from towers or a slot number)")
            continue
        if f"PLACE_TOWER_{step['slot']}" not in actions:
            problems.append(f"plan step {i}: add PLACE_TOWER_{step['slot']} to actions to place {step['tower']}")
        if not isinstance(step["count"], int) or step["count"] < 0:
            problems.append(f"plan step {i}: count must be 0 (upgrade the ones placed) or more")
        if not isinstance(step["upgrade_to"], int) or step["upgrade_to"] < 0:
            problems.append(f"plan step {i}: upgrade_to must be 0 or more")
    if cfg.get("after_plan", "ai") not in ("ai", "upgrade"):
        problems.append('after_plan must be "ai" or "upgrade"')
    clash = [k for name, k in keys.items() if k.upper() in hotkeys and name != "cancel"]  # cancel: see OWN_PRESSES
    if clash:
        problems.append(f"game keys {clash} collide with the pause/kill hotkeys")
    if problems:
        raise SystemExit("config.json problems:\n  - " + "\n  - ".join(problems))
    if not cfg.get("placement_spots") and any(a.startswith("PLACE_TOWER_") for a in actions):
        print("WARNING: no placement_spots configured - tower placement is disabled. Run calibrate.py.")


class GameWindow:
    """Tracks the Roblox client area; falls back to the fixed region from config."""

    def __init__(self, capture_cfg):
        self.title = capture_cfg.get("window_title", "Roblox")
        self.auto = capture_cfg.get("auto_find_window", True)
        self.fallback = dict(capture_cfg["region"])
        self.hwnd = None
        self.found = False
        self.region = self.fallback

    def refresh(self):
        if not self.auto:
            return
        if not (self.hwnd and user32.IsWindow(self.hwnd)):
            self.hwnd = find_window(self.title)
        region = client_region(self.hwnd)
        self.found = region is not None
        self.region = region or self.fallback

    def focus(self):
        return focus_window(self.hwnd)

    def is_focused(self):
        fg = user32.GetForegroundWindow()
        if self.hwnd:
            return fg == self.hwnd
        return bool(fg) and window_title(fg) == self.title


class Control:
    """Pause/kill flags shared by every thread."""

    def __init__(self, start_paused=True):
        self.stop = threading.Event()
        self._running = threading.Event()
        if not start_paused:
            self._running.set()

    @property
    def paused(self):
        return not self._running.is_set()

    def toggle_pause(self):
        if self.stop.is_set():
            return
        if self._running.is_set():
            self._running.clear()
            print("[failsafe] PAUSED")
        else:
            self._running.set()
            print("[failsafe] RUNNING")

    def resume(self):
        if not self.stop.is_set() and not self._running.is_set():
            self._running.set()
            print("[bot] RUNNING")

    def kill(self):
        self._running.clear()
        if not self.stop.is_set():
            self.stop.set()
            print("[failsafe] KILLED - shutting down")

    def should_abort(self):
        return self.stop.is_set() or not self._running.is_set()


class Failsafe(threading.Thread):
    """Polls the physical keyboard, so the hotkeys work while Roblox has focus."""

    def __init__(self, pause_key, kill_key, on_pause, on_kill):
        super().__init__(name="failsafe", daemon=True)
        self.bindings = [(vk_code(pause_key), on_pause), (vk_code(kill_key), on_kill)]
        self._halt = threading.Event()

    def run(self):
        was_down = {}
        while not self._halt.is_set():
            for vk, callback in self.bindings:
                down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
                if down and not was_down.get(vk) and time.monotonic() >= OWN_PRESSES.get(vk, 0):
                    callback()
                was_down[vk] = down
            time.sleep(0.01)

    def halt(self):
        self._halt.set()


class Aborted(Exception):
    """Raised mid-action when the bot is paused, killed, or Roblox loses focus."""


class PlacementBlocked(Exception):
    """No position around the spot accepted the tower."""


@dataclass
class Goal:
    """What the bot must do before anything else: "place" a tower slot, "upgrade" a tower,
    or "hold" (no placements, upgrades allowed)."""
    kind: str
    target: object
    note: str
    step: int | None = None  # build-plan step it comes from


def match_loadout(prices, towers):
    """Which configured tower sits in each hotbar slot, judged by price; None if it doesn't add up.

    A price identifies its tower. Two towers with the same price (Minigunner and Warden, $1,850)
    can't be told apart, so for them the configured slot is kept. A read like 11850 still counts as
    $1,850 (the slot number in the corner sometimes gets read into the price). Returns {slot: config slot}.
    """
    costs = {slot: t.get("cost", 0) for slot, t in towers.items() if t.get("cost")}
    mapping = {}
    for i, price in enumerate(prices, 1):
        if price <= 0:
            continue
        hits = [s for s, c in costs.items() if price == c] or \
               [s for s, c in costs.items() if str(price).endswith(str(c))]
        if len(hits) > 1 and str(i) in hits:
            hits = [str(i)]  # same price as another tower: assume it is where config.json says
        if len(hits) != 1 or hits[0] in mapping.values():
            return None
        mapping[str(i)] = hits[0]
    return mapping if len(mapping) == len(costs) else None


def apply_loadout(cfg, mapping):
    """Re-key towers (and the PLACE_TOWER actions) so slot N is the tower really in hotbar slot N.

    Changes cfg["towers"] and cfg["actions"] in place, since the brain and executor share them.
    """
    old = dict(cfg["towers"])
    cfg["towers"].clear()
    cfg["towers"].update({slot: old[src] for slot, src in mapping.items()})
    back = {src: slot for slot, src in mapping.items()}
    cfg["actions"][:] = [f"PLACE_TOWER_{back[a[-1]]}" if a.startswith("PLACE_TOWER_") and a[-1] in back else a
                         for a in cfg["actions"]]


def parse_plan(cfg):
    """The build plan from config as [{"slot", "count", "upgrade_to"}]; towers by name or slot number."""
    towers = cfg.get("towers", {})
    by_name = {t.get("name", "").lower(): slot for slot, t in towers.items()}
    plan = []
    for step in cfg.get("plan") or []:
        key = str(step.get("tower", "")).strip()
        slot = key if key in towers else by_name.get(key.lower())
        plan.append({"slot": slot, "tower": key, "count": step.get("count", 1),
                     "upgrade_to": step.get("upgrade_to", 0)})
    return plan


class RobloxInput:
    """DirectInput (scan-code) mouse and keys, which Roblox registers unlike virtual-key events."""

    def __init__(self, input_cfg, should_abort):
        pdi.PAUSE = 0          # every step is timed below instead
        pdi.FAILSAFE = True    # slamming the mouse into the top-left corner also stops the bot
        self.key_hold = input_cfg.get("key_hold", 0.04)
        self.click_hold = input_cfg.get("click_hold", 0.05)
        self.step = input_cfg.get("step_delay", 0.06)
        self.should_abort = should_abort

    def _guard(self):
        if self.should_abort():
            raise Aborted

    def wait(self, seconds=None):
        time.sleep(self.step if seconds is None else seconds)
        self._guard()

    def move(self, x, y):
        self._guard()
        pdi.moveTo(x, y)
        # Roblox only notices the cursor after a real movement event, so nudge it.
        pdi.moveTo(x + 1, y)
        pdi.moveTo(x, y)

    def press(self, key):
        self._guard()
        try:  # the bot's own Q (TDS: cancel placement) must not trip the Q pause hotkey
            OWN_PRESSES[vk_code(key)] = time.monotonic() + self.key_hold + 0.2
        except ValueError:
            pass
        pdi.keyDown(key)
        try:
            time.sleep(self.key_hold)
        finally:
            pdi.keyUp(key)

    def click(self, x, y):
        self.move(x, y)
        self.wait()
        pdi.mouseDown()
        try:
            time.sleep(self.click_hold)
        finally:
            pdi.mouseUp()


class DryRunInput:
    """Stand-in for RobloxInput that never touches the mouse or keyboard."""

    step = 0.0

    def wait(self, seconds=None):
        pass

    def move(self, x, y):
        pass

    def press(self, key):
        pass

    def click(self, x, y):
        pass


class ActionExecutor:
    """Vets decisions against cooldowns and what the bot has done so far, then performs them."""

    def __init__(self, cfg, control, window, get_region, dry_run):
        self.actions = cfg["actions"]
        self.spots = [tuple(s) for s in cfg.get("placement_spots", [])]
        self._base_spots = list(self.spots)  # as calibrated; the bot moves and adds spots during a match
        self.towers = cfg.get("towers", {})
        self.keys = cfg["keys"]
        self.cancel_key = self.keys.get("cancel", "q")  # TDS: Q cancels placing; does nothing with empty hands
        self.maybe_holding = False  # a tower key was pressed and the hand not emptied since
        self.tried = {}  # ground spot -> the point last clicked for it (fractions)
        self.bad_points = []  # points (fractions) where a ground tower didn't appear
        self.cooldowns = cfg.get("cooldowns", {})
        self.skip_button = cfg.get("skip_wave_button")
        self.require_focus = cfg["safety"].get("require_focus", True)
        self.dry_run = dry_run
        self.control = control
        self.window = window
        self.get_region = get_region
        self.input = DryRunInput() if dry_run else RobloxInput(cfg.get("input", {}), self._should_abort)
        self.placed = []      # [{"slot": "1", "spot": 3, "upgrades": 0}, ...] in placement order
        self.selected = None  # the tower (dict from self.placed) the game has selected
        self.reader_on = False  # set by the bot when exact numbers (incl. the upgrade panel) are read
        self.cash = -1        # latest cash minus spending the reader can't show yet
        self.spent = deque()  # (time, amount, cash believed before) of recent placements and upgrades
        self.inspect = None   # tower whose upgrade panel should be read next
        self.plan = parse_plan(cfg)
        self.after_plan = cfg.get("after_plan", "ai")  # "ai": the AI plays on; "upgrade": only upgrades
        self.auto_ready = cfg.get("auto_ready", True)
        self.auto_restart = cfg.get("auto_restart", True)  # click Restart Match when a match ends
        self.abilities = cfg.get("abilities", [])  # tower abilities with keys, e.g. Commander: F, B
        handover = cfg.get("handover", {})
        self.handover_wave = handover.get("wave", 0)     # from this wave the player plays, the bot records (0: never)
        self.handover_wait_s = handover.get("nobody_s", 45)  # nobody touched the game by then: the bot plays on
        self.handover_beep = handover.get("beep", True)
        self.recorder = None         # recorder.HumanRecorder, set by the bot: the player's input and the recording
        self.endgame = None          # endgame.Endgame, set by the bot: plays waves 34-35 the way the player won
        self.forced_goal = None      # a Goal the endgame is carrying out right now (overrides the plan)
        self.place_target = None     # map point the next placement goes to (endgame leapfrog), else a spot
        self.human_turn = False      # the player is playing; the bot only watches
        self._handed_over = self._human_played = False
        self._handover_at = 0.0
        self._ability_at = {}
        self.camera = CameraTracker()
        self._restart_at = float("-inf")
        self._end_seen, self._end_recorded = 0, False
        self.hold_until = 0.0  # no actions until then: a new match is loading
        self.matches = 0
        self.match_started = time.time()
        self.match_log = cfg.get("match_log", "matches.jsonl")
        self._best_wave = -1
        self.wave = -1        # the wave the bot acts on: a read wave number only once it makes sense (_confirm_wave)
        self._wave_cand, self._wave_hits, self._wave_seen_t = None, 0, None
        self.ready_roi = cfg.get("rois", {}).get("ready_button")
        self._ready_at = float("-inf")
        self.history = deque(maxlen=6)
        self.reading = None   # latest exact numbers (reader.Reading), passed in through observe()
        self.pending = None   # the last placement, until the game's tower counter confirms it
        self.failed_spots = Counter()
        self.prober = Prober(cfg)
        placement = cfg.get("placement", {})
        self.auto_spots = placement.get("auto_spots", True)
        self.auto_spot_px = placement.get("auto_spot_px", 70)
        self._grown = set()   # spots that already have automatic neighbours
        self.reserved = {}    # spot index -> tower slot that found it (cliff spots)
        self._cliff_retry = {}  # tower slot -> time to search for a cliff again
        self.step_fails = Counter()   # plan step -> failed placements in a row
        self.step_skip_until = {}     # plan step -> time until which the plan skips it
        self.cliff_search_s = placement.get("cliff_search_s", 10.0)
        self.cluster_px = placement.get("cluster_px", 250)  # how far from the first one a clustered tower may go
        self._next_upgrade = 0
        self._last_used = {}

    def _should_abort(self):
        return self.control.should_abort() or (self.require_focus and not self.window.is_focused())

    @staticmethod
    def _group(action):
        return "PLACE_TOWER" if action.startswith("PLACE_TOWER_") else action

    def _free_spot(self, slot=None):
        """First free spot from the tower's own `spots` list (1-based numbers from the preview), in its
        order; after that (or without a list) the first free spot no other tower type lists, growing
        new spots around placed towers when none is left."""
        allowed = self.towers.get(slot, {}).get("spots") if slot else None
        found = self._first_free(allowed, slot) if allowed else None
        if found is None:
            found = self._first_free(self._unclaimed(slot), slot)
        if found is None and self.auto_spots and self._grow_spots():
            found = self._first_free(self._unclaimed(slot), slot)
        return found

    def _unclaimed(self, slot):
        """Spot numbers no other tower type has in its `spots` list. A melee (`hug_path`) tower takes the
        ones nearest its kind first: those spots grew around the others, right by the path."""
        claimed = {n for s, t in self.towers.items() if s != slot for n in t.get("spots") or []}
        free = [n for n in range(1, len(self.spots) + 1) if n not in claimed]
        mine = [self.spots[t["spot"]] for t in self.placed if t["slot"] == slot]
        if mine and self.towers.get(slot, {}).get("hug_path"):
            free.sort(key=lambda n: min(math.hypot((self.spots[n - 1][0] - x) * 16, (self.spots[n - 1][1] - y) * 9)
                                        for x, y in mine))
        return free


    def _first_free(self, allowed, slot=None):
        """Failures count per tower type (a cliff that rejects a ground tower may suit a Ranger), and
        spots a cliff tower found are kept for that tower type."""
        used = {t["spot"] for t in self.placed}
        candidates = [n - 1 for n in allowed] if allowed is not None else range(len(self.spots))
        return next((i for i in candidates if 0 <= i < len(self.spots) and i not in used
                     and self.reserved.get(i, slot) == slot
                     and self.failed_spots[(i, slot)] < MAX_SPOT_FAILURES), None)

    def _tower_boxes(self, region, slots=None, r_px=40):
        """Screen boxes around towers the bot placed (all, or of these slots): Farms get no red ring
        in the game's shading, so the map alone would see their ground as clear."""
        r = round(r_px * region["width"] / 1920)
        boxes = []
        for t in self.placed:
            if slots is None or t["slot"] in slots:
                x, y = self._to_screen(self.spots[t["spot"]], region)
                boxes.append((x - r, y - r, 2 * r, 2 * r))
        return boxes

    def _failed_here(self, spot, region):
        """Screen boxes around the points where a ground tower (of any type) failed to appear, so the
        next try picks a different clear point: the shading shows the path, not scenery such as the
        barrels or the platform in the middle of the map."""
        r = round(40 * region["width"] / 1920)
        boxes = []
        for p in self.bad_points:
            x, y = self._to_screen(p, region)
            boxes.append((x - r, y - r, 2 * r, 2 * r))
        return boxes

    def _bad_cliffs(self, slot, region):
        """Screen boxes around cliff spots where this tower already failed to appear."""
        r = round(45 * region["width"] / 1920)
        boxes = []
        for (spot, s), n in self.failed_spots.items():
            if s == slot and n and spot < len(self.spots):
                x, y = self._to_screen(self.spots[spot], region)
                boxes.append((x - r, y - r, 2 * r, 2 * r))
        return boxes

    def _is_cliff(self, slot):
        return bool(self.towers.get(slot, {}).get("cliff"))

    def _cliff_points(self, slot, region):
        """Where a cliff-only tower (Ranger) might go: around its own towers first, then a grid over the map."""
        no_go = self.prober.no_go + ([self.prober.panel] if self.prober.panel else [])
        taken = [self.spots[t["spot"]] for t in self.placed]
        near = []
        for t in self.placed:
            if t["slot"] == slot:
                fx, fy = self.spots[t["spot"]]
                near += [(fx + math.cos(2 * math.pi * k / 8) * r / 1920, fy + math.sin(2 * math.pi * k / 8) * r / 1080)
                         for r in (60, 95) for k in range(8)]
        grid = [(x / 100, y / 100) for y in range(10, 91, 9) for x in range(6, 96, 9)]

        def usable(p):
            return (0.05 < p[0] < 0.95 and 0.05 < p[1] < 0.95
                    and not any(x <= p[0] <= x + w and y <= p[1] <= y + h for x, y, w, h in no_go)
                    and all(math.hypot((p[0] - q[0]) * 1920, (p[1] - q[1]) * 1080) >= 45 for q in taken))

        return [self._to_screen(p, region) for p in near + grid if usable(p)]

    def _grow_spots(self):
        """Out of spots: add new ones around the towers that really got placed (they sit near the path)."""
        unconfirmed = self.pending["tower"] if self.pending else None
        no_go = self.prober.no_go + ([self.prober.panel] if self.prober.panel else [])
        added = False
        for tower in self.placed:
            if (tower is unconfirmed or tower["spot"] in self._grown or len(self.spots) >= MAX_SPOTS
                    or self._is_cliff(tower["slot"])):  # around a cliff tower is cliff, not ground
                continue
            self._grown.add(tower["spot"])
            fx, fy = self.spots[tower["spot"]]
            for k in range(8):
                a = 2 * math.pi * k / 8
                gx = fx + math.cos(a) * self.auto_spot_px / 1920
                gy = fy + math.sin(a) * self.auto_spot_px / 1080
                if not (0.05 < gx < 0.95 and 0.05 < gy < 0.95):
                    continue
                if any(x <= gx <= x + w and y <= gy <= y + h for x, y, w, h in no_go):
                    continue
                if any(math.hypot((gx - sx) * 1920, (gy - sy) * 1080) < 0.6 * self.auto_spot_px for sx, sy in self.spots):
                    continue
                self.spots.append((gx, gy))
                added = True
        return added

    def _parks(self, region):
        """Where the cursor waits while the shading is read: the left and right edges, far apart, so the
        held tower's range circle and menu (which follow the cursor) never cover the same place twice."""
        return [self._to_screen((0.03, 0.5), region), self._to_screen((0.97, 0.5), region)]

    def observe(self, reading):
        """Take in the latest numbers: cash minus spending the reading can't show yet, the upgrade
        panel of the tower the bot clicked, and whether the last placement really happened."""
        self.reading = reading
        self._confirm_wave(reading)
        self._best_wave = max(self._best_wave, self.wave)
        now = time.monotonic()
        while self.spent and now - self.spent[0][0] > 15:
            self.spent.popleft()
        self.cash = reading.cash - self._unseen_spending(reading, now) if reading.cash >= 0 else -1
        notes = [n for n in (self._handover(reading), self._check_panel(reading), self._check_placement(reading)) if n]
        return "; ".join(notes) or None

    def _confirm_wave(self, reading):
        """Set self.wave from the reader, but only when the number makes sense. One bad reading (898,
        797 or 880 around wave 20, cash read as 5) started the endgame at wave 20 in 4 of 10 overnight
        matches: it sold the Farms, stopped building and lost at wave 28-30. Waves go up one at a time,
        so the next wave needs 2 separate readings that agree; any other jump (the bot was paused, a
        new match it didn't see start) needs 5, and an unknown wave 3."""
        if reading.t and reading.t == self._wave_seen_t:
            return  # the same reading as last round: it counts once
        self._wave_seen_t = reading.t
        w, total = reading.wave, reading.total_waves
        if not 0 <= w <= MAX_WAVE or (0 < total <= MAX_WAVE and w > total):
            return
        if w == self.wave:
            self._wave_cand, self._wave_hits = None, 0
            return
        self._wave_hits = self._wave_hits + 1 if w == self._wave_cand else 1
        self._wave_cand = w
        need = (2 if w == self.wave + 1 else 5) if self.wave >= 0 else 3
        if self._wave_hits >= need:
            self.wave, self._wave_cand, self._wave_hits = w, None, 0

    def _handover(self, reading):
        """From wave `handover.wave` on, the player plays and the bot records how (recorder.py). If
        nobody clicks or presses a game key in Roblox within `handover.nobody_s`, the bot plays the
        wave itself, so an unattended run still finishes its matches."""
        if not self.handover_wave:
            return None
        now = time.monotonic()
        if self.human_turn:
            if self.recorder and self.recorder.last_input > self._handover_at + 0.5:
                self._human_played = True
            if self._human_played or now - self._handover_at < self.handover_wait_s:
                return None
            self.human_turn = False
            if self.recorder:
                self.recorder.stop_recording("nobody took over; the bot played on")
            return f"nobody took over in {self.handover_wait_s:.0f} s: the bot plays wave {self.wave} itself"
        if self._handed_over or self._end_seen or now < self.hold_until:
            return None
        if self.wave < self.handover_wave:  # the confirmed wave: never on one misread number
            return None
        self.human_turn = self._handed_over = True
        self._human_played, self._handover_at = False, now
        if self.maybe_holding and not self.dry_run:
            try:
                self._put_back()  # don't leave a tower in the player's hand
            except Aborted:
                pass
        if self.handover_beep:
            threading.Thread(target=_beep, daemon=True).start()
        where = ""
        if self.recorder:
            towers = [{"tower": self.tower_name(t["slot"]), "level": t["upgrades"],
                       "map": [round(v, 4) for v in self.spots[t["spot"]]] if t["spot"] < len(self.spots) else None}
                      for t in self.placed]
            folder = self.recorder.start_recording(self.wave, towers)
            where = f" in recordings/{folder.name}" if folder else ""
        return (f"WAVE {self.wave}: YOUR TURN. The bot stops and records how you play{where}. "
                f"Nobody playing within {self.handover_wait_s:.0f} s: the bot plays on")

    def _unseen_spending(self, reading, now):
        """Money spent that this reading doesn't show yet: spent after its frame, or spent just before
        it while the game's cash display hadn't caught up (it lags a moment behind a purchase)."""
        unseen = 0
        for t, amount, before in self.spent:
            if t > reading.t:
                unseen += amount
            elif now - t < 3 and before >= 0 and reading.cash > before - amount / 2:
                unseen += amount  # the frame is newer, but the cash shown is still the old amount
        return unseen

    def _check_panel(self, reading):
        """After the bot clicked a tower or upgraded it: read what its upgrade panel says."""
        insp = self.inspect
        if insp is None or reading.t <= insp["t"] + PANEL_SETTLE_S:
            return None
        self.inspect = None
        tower = insp["tower"]
        if not any(t is tower for t in self.placed):
            return None
        name, spot = self.tower_name(tower["slot"]), tower["spot"] + 1
        if reading.upgrade_cost < 0:  # clicked the tower, but no upgrade panel showed up
            self.selected = None
            tower["misses"] = tower.get("misses", 0) + 1
            # Forget it only when the game's counter agrees the bot has more towers than exist.
            if tower["misses"] >= 2 and 0 <= reading.towers < len(self.placed):
                self._forget(tower)
                return f"no {name} at spot {spot} (no upgrade panel, tower counter {reading.towers}) - forgotten"
            if tower["misses"] >= 3:
                tower["misses"], tower["rest_until"] = 0, time.monotonic() + 30
                return f"can't open {name}'s upgrade panel at spot {spot}; leaving it for 30 s"
            return None
        tower["misses"] = 0
        costs = self.towers.get(tower["slot"], {}).get("upgrade_costs")
        if reading.upgrade_cost == 0 and reading.upgrade_maxed:
            tower["maxed"] = True
            top = self.towers.get(tower["slot"], {}).get("max_level") or len(costs or [])
            tower["upgrades"] = max(tower["upgrades"], top)  # abilities check the level (Caravan: 4)
            return f"{name} at spot {spot} is fully upgraded"
        if reading.upgrade_cost == 0 and costs and tower["upgrades"] < len(costs):
            # An affordable upgrade turns the price box green, and the model then reads "Fully Upgraded".
            # The level bars aren't all lit, so it isn't: use the known price for its level.
            tower["zero_reads"] = tower["wrong_panels"] = 0
            tower["next_cost"] = costs[tower["upgrades"]]
            return None
        if reading.upgrade_cost == 0:
            # A price the reader can't see (the Commander's message box covers the panel for a few
            # seconds) also reads as 0. Below the tower's known top level, look twice more first.
            top = self.towers.get(tower["slot"], {}).get("max_level") or len(self.towers.get(tower["slot"], {}).get("upgrade_costs") or [])
            tower["zero_reads"] = tower.get("zero_reads", 0) + 1
            if not (top and tower["upgrades"] >= top) and tower["zero_reads"] < (3 if top else 2):
                self.selected = None
                tower["rest_until"] = time.monotonic() + 5
                return f"{name} at spot {spot} looks fully upgraded (level {tower['upgrades']}); checking again"
            tower["maxed"] = True
            return f"{name} at spot {spot} is fully upgraded"
        tower["zero_reads"] = 0
        note = None
        if costs and reading.upgrade_cost in costs:
            # Each price belongs to one level, so the panel tells the tower's real level; the bot's own
            # count drifts when a press doesn't take or a click opens a neighbour.
            level = costs.index(reading.upgrade_cost)
            if level != tower["upgrades"]:
                note = f"{name} at spot {spot} is level {level} (by its price), not {tower['upgrades']}"
                tower["upgrades"] = level
        elif costs and any(reading.upgrade_cost in (t.get("upgrade_costs") or []) for s, t in self.towers.items()
                           if s != tower["slot"]):
            self.selected = None  # that price belongs to another tower type: the click opened the wrong tower
            tower["wrong_panels"] = tower.get("wrong_panels", 0) + 1
            if tower["wrong_panels"] >= 3:
                tower["wrong_panels"], tower["rest_until"] = 0, time.monotonic() + 60
                return f"{name} at spot {spot}: clicking it keeps opening another tower; leaving it for 60 s"
            tower["rest_until"] = time.monotonic() + 3
            return f"{name} at spot {spot}: the panel shows another tower's price (${reading.upgrade_cost:,}); trying again"
        tower["wrong_panels"] = 0
        tower["next_cost"] = reading.upgrade_cost
        return note

    def press_ready(self, frame, region):
        """Click the green READY button that TDS shows before the first wave."""
        if not self.auto_ready or self.ready_roi is None or time.monotonic() - self._ready_at < 6:
            return None  # one click, then give the game time; clicking again could un-ready
        if not ready_button_visible(frame, self.ready_roi):
            return None
        self._ready_at = time.monotonic()
        if self.dry_run:
            return "[dry-run] would click READY"
        if self.require_focus and not self.window.is_focused():
            return None
        x, y, w, h = self.ready_roi
        try:
            self.input.click(*self._to_screen((x + w / 2, y + h / 2), region, ui=True))
        except Aborted:
            return None
        return "clicked READY"

    def press_restart(self, frame, region):
        """At the end-of-match screen, note how far the match got and click Restart Match; the new
        match then starts with the READY button, which press_ready clicks."""
        if not self.auto_restart:
            return None
        point = end_screen_button(frame)
        if point is None:
            self._end_seen, self._end_recorded = 0, False
            return None
        self._end_seen += 1  # while it shows, the bot does nothing else (see vet)
        # Seen twice in a row (never a stray green blob). The new match loads within ~3 s of a click
        # that took, so the screen still up 8 s later means the click didn't take: click again.
        if self._end_seen < 2 or time.monotonic() - self._restart_at < 8:
            return None
        note = None
        if not self._end_recorded:
            note, self._end_recorded = self._record_match(self._best_wave, lost_title(frame, point)), True
            if self.recorder and self.recorder.recording:
                self.recorder.stop_recording(note)
        self._restart_at = time.monotonic()
        if self.dry_run:
            return f"[dry-run] {note}; would click Restart Match"
        if self.require_focus and not self.window.is_focused():
            return None
        try:
            self.input.click(*self._to_screen(point, region, ui=True))
        except Aborted:
            return None
        self.new_match()
        return f"{note}; clicked Restart Match" if note else "clicked Restart Match again (the result screen was still up)"

    def _record_match(self, wave, lost=True):
        self.matches += 1
        minutes = (time.time() - self.match_started) / 60
        total = self.reading.total_waves if self.reading else -1
        won = not lost and total > 0 and wave >= total  # the title wasn't the red YOU LOST
        if self.match_log:
            try:
                with open(Path(__file__).resolve().parent / self.match_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ended": time.strftime("%Y-%m-%d %H:%M:%S"), "match": self.matches,
                                        "wave": wave, "total_waves": total, "won": won,
                                        "minutes": round(minutes, 1), "towers": len(self.placed),
                                        "player_played": self._human_played}) + "\n")
            except OSError:
                pass
        return f"match {self.matches} over at wave {wave} after {minutes:.0f} min" + (" - WON" if won else "")

    def new_match(self):
        """Forget the last match: its towers, spots moved or added, failures, blocked points and plan
        progress. Then hold still for a few seconds while the new match loads."""
        self.spots = list(self._base_spots)
        self.placed, self.selected, self.pending, self.inspect = [], None, None, None
        self.spent.clear()
        self.cash = -1
        self.history.clear()
        self.failed_spots.clear()
        self.reserved, self.tried, self._cliff_retry = {}, {}, {}
        self._grown = set()
        self.step_fails.clear()
        self.step_skip_until = {}
        self._next_upgrade = 0
        self._last_used = {}
        self.maybe_holding = False
        self._ready_at = float("-inf")
        self._best_wave = -1
        self.wave, self._wave_cand, self._wave_hits = -1, None, 0
        self.match_started = time.time()
        self.bad_points = []  # a wrong one blocks a good spot for good; kept across matches they piled up
        self.camera.reset()
        self._ability_at = {}
        if self.recorder and self.recorder.recording:
            self.recorder.stop_recording("a new match started")
        self.human_turn = self._handed_over = self._human_played = False
        self.forced_goal = self.place_target = None
        if self.endgame is not None:
            self.endgame.reset()
        self.hold_until = time.monotonic() + 8

    def _under_panel(self, point):
        """Is this spot (fractions) where the upgrade panel shows when it is open?"""
        box = self.prober.panel
        if not box:
            return False
        x, y, w, h = box
        return x - 0.02 <= point[0] <= x + w + 0.02 and y - 0.03 <= point[1] <= y + h + 0.02

    def _panel_visible(self, region):
        """Look at the screen: is a tower's upgrade panel open (its red Sell button showing)?"""
        if self.prober.panel is None:
            return self.selected is not None
        x, y, w, h = self.prober.panel
        box = {"left": round(region["left"] + x * region["width"]), "top": round(region["top"] + y * region["height"]),
               "width": max(1, round(w * region["width"])), "height": max(1, round(h * region["height"]))}
        return sell_button_visible(self.prober.grab(box))

    def _close_panel(self, region):
        """Close the upgrade panel by clicking empty ground; it covers the lower placement spots.

        Clicks the point farthest from every tower (free spots and a grid over the map, outside the
        HUD and the panel), checks that the Sell button is gone, and tries the next point if not.
        """
        if not self._panel_visible(region):
            self.selected = None
            return True
        self._empty_hand()
        towers = [self.spots[t["spot"]] for t in self.placed]
        no_go = self.prober.no_go + ([self.prober.panel] if self.prober.panel else [])

        def gap(p):  # pixels (at 1920x1080) to the nearest tower
            return min((math.hypot((p[0] - q[0]) * 1920, (p[1] - q[1]) * 1080) for q in towers), default=1e9)

        grid = [(gx / 100, gy / 100) for gx in range(12, 90, 12) for gy in range(22, 86, 12)]
        options = [p for p in list(self.spots) + grid if gap(p) >= 90
                   and not any(x <= p[0] <= x + w and y <= p[1] <= y + h for x, y, w, h in no_go)]
        for p in sorted(options, key=gap, reverse=True)[:3]:
            self.input.click(*self._to_screen(p, region))
            self.input.wait(0.2)
            if not self._panel_visible(region):
                self.selected, self.inspect = None, None
                return True
        return False

    def _forget(self, tower):
        self.placed = [t for t in self.placed if t is not tower]
        if self.selected is tower:
            self.selected = None

    def _check_placement(self, reading):
        """Check the last placement against the game's tower counter; undo it if the counter never moved."""
        p = self.pending
        if p is None:
            return None
        age = time.monotonic() - p["t"]
        newer = reading.t > p["t"] + 0.2  # read from a frame captured after the click
        if p["before"] < 0 or (newer and reading.towers < 0) or age > 3 * PLACE_CONFIRM_S:
            self.pending = None  # no usable counter: trust the click
            return None
        if newer and 0 <= reading.towers < p["before"]:
            self.pending = None  # the counter went down: a new match or a misread, nobody's fault
            return None
        if newer and reading.towers > p["before"]:
            self.pending = None  # confirmed
            self.step_fails.pop(p["tower"].get("step"), None)
            return None
        if newer and age > PLACE_CONFIRM_S:
            self.pending = None
            tower = p["tower"]
            self._forget(tower)
            key = (tower["spot"], tower["slot"])
            self.failed_spots[key] += 1
            dropped = " - spot dropped" if self.failed_spots[key] >= MAX_SPOT_FAILURES else ""
            self.prober.save_failure(f"{self.tower_name(tower['slot'])}_spot{tower['spot'] + 1}".replace(" ", ""))
            if not self._is_cliff(tower["slot"]):  # scenery stays put: no ground tower tries that point again
                self.bad_points.append(self.tried.get(tower["spot"], self.spots[tower["spot"]]))
            skipped = self._step_failed(tower.get("step"))
            return (f"{self.tower_name(tower['slot'])} did not appear at spot {tower['spot'] + 1} "
                    f"(tower counter stayed at {p['before']}){dropped}" + (f"; {skipped}" if skipped else ""))
        return None

    def _step_failed(self, step):
        """A plan step's tower failed to go down; after 3 failures the plan skips that step for 2 minutes,
        so one impossible tower can't freeze the whole defense."""
        if step is None:
            return None
        self.step_fails[step] += 1
        if self.step_fails[step] < 3:
            return None
        self.step_fails[step] = 0
        self.step_skip_until[step] = time.monotonic() + 120
        return f"plan step {step + 1} skipped for 2 minutes (its tower keeps failing)"

    def tower_name(self, slot):
        return self.towers.get(slot, {}).get("name", f"slot {slot} tower")

    def _to_screen(self, point, region, ui=False):
        """A map point (fractions as calibrated) on screen now, following the camera if a boss skill
        moved or turned it; `ui` points (buttons) stay put."""
        fx, fy = point if ui else self.camera.to_screen(*point)
        return round(region["left"] + fx * region["width"]), round(region["top"] + fy * region["height"])

    def _from_screen(self, x, y, region):
        return self.camera.to_map((x - region["left"]) / region["width"], (y - region["top"]) / region["height"])

    def track_camera(self, frame):
        if self.maybe_holding:
            return  # holding a tower turns the path red: not the view moving
        in_match = (self.reading is not None and self.reading.wave >= 1 and not self._end_seen
                    and time.monotonic() >= self.hold_until)
        self.camera.update(frame, in_match)

    def use_abilities(self, frame, region):
        """Press tower abilities that have keys (Commander: Call to Arms F, Support Caravan B) when they
        are ready: from their tower's level, every cooldown (shared out over the towers that have it),
        the costly ones while a boss health bar is up (or from a late wave on) and the cash covers them."""
        if not self.abilities or self._end_seen or self.human_turn or time.monotonic() < self.hold_until:
            return None
        if self.endgame is not None and self.endgame.giving_up:
            return None  # wave 35 is lost on purpose: nothing to spend on
        r = self.reading
        if r is None or self.wave < 1:
            return None
        now, notes, boss = time.monotonic(), [], None
        for ab in self.abilities:
            name = ab["name"]
            owners = [t for t in self.placed
                      if self.tower_name(t["slot"]) == ab.get("tower") and t["upgrades"] >= ab.get("min_level", 0)]
            if not owners or self.wave < ab.get("from_wave", 1):
                continue
            cooldown = ab.get("cooldown", 30) / (len(owners) if ab.get("per_tower") else 1)
            if now - self._ability_at.get(name, float("-inf")) < max(3.0, cooldown):
                continue
            cost = ab.get("cost", 0)
            if cost and not 0 <= cost + ab.get("keep", 0) <= self.cash:
                continue
            if ab.get("when") == "boss":
                boss = boss_bar_visible(frame) if boss is None else boss
                if not boss and self.wave < ab.get("always_from_wave", 10 ** 6):
                    continue
            if self.dry_run:
                notes.append(f"[dry-run] would use {name}")
                self._ability_at[name] = now
                continue
            if self.require_focus and not self.window.is_focused():
                return None
            try:
                self.input.press(ab["key"])
            except Aborted:
                return None
            self._ability_at[name] = now
            if cost:
                self.spent.append((now, cost, self.cash))
            notes.append(f"used {name} ({ab['key'].upper()})")
        return "; ".join(notes) or None

    def context(self):
        """Bot-side state the AI cannot see on screen."""
        now = time.monotonic()
        counts = Counter(self.tower_name(t["slot"]) for t in self.placed)
        return {
            "placed": len(self.placed),
            "spots": len(self.spots),
            "upgrades": sum(t["upgrades"] for t in self.placed),
            "towers": ", ".join(f"{name} x{n}" for name, n in counts.items()) or "none",
            "history": [f"{a} ({now - t:.0f}s ago)" for t, a in self.history],
            "feasible": [a for a in self.actions if self._blocked(a) is None],
            "saving": goal.note if (goal := self._goal()) else "",
            "must": bool(goal and goal.kind in ("place", "upgrade")),  # the plan says do it: no point waiting
        }

    def _blocked(self, a):
        """Why an action is impossible in the current bot state (cash and cooldowns aside), or None."""
        goal = self._goal()
        if a.startswith("PLACE_TOWER_"):
            blocked = self._placement_blocked(a[-1])
            if blocked:
                return blocked
            if goal and not (goal.kind == "place" and goal.target == a[-1]):
                return goal.note
        elif a in ("UPGRADE_SELECTED", "SELL") and not self.placed:
            return f"{a}: no towers placed yet"
        elif a == "UPGRADE_SELECTED" and goal and goal.kind == "upgrade":
            return None if self._upgradable(goal.target) else goal.note
        elif a == "UPGRADE_SELECTED" and goal and goal.kind == "place":
            return goal.note
        elif a == "UPGRADE_SELECTED" and not any(self._upgradable(t) for t in self.placed):
            return "nothing to upgrade: every tower is fully upgraded or its next upgrade costs more than the cash"
        return None

    def _goal(self):
        """What the bot must do before anything else, or None to let the AI choose freely.

        With a build plan in config, the plan decides. Without one, a tower's `upgrade_to`
        and `save_for` settings do. The endgame's current goal comes before both.
        """
        if self.forced_goal is not None:
            return self.forced_goal
        if self.plan:
            return self._plan_goal()
        focus = self._focus_tower()
        if focus:
            return Goal("upgrade", focus, self._focus_note(focus))
        saving = self._saving_for()
        if saving:
            return Goal("place", saving, self._saving_note(saving))
        return None

    def _plan_goal(self):
        """Walk the plan in order: the first step that isn't finished gives the goal.

        A step places `count` towers one by one and upgrades each to `upgrade_to` before the next.
        A step whose tower can't be placed (its limit, no free spot, its waves are over) is
        skipped; one waiting for its first wave holds placements but lets upgrades happen.
        """
        n = len(self.plan)
        for i, step in enumerate(self.plan):
            if time.monotonic() < self.step_skip_until.get(i, 0):
                continue  # its tower kept failing; come back to it later
            slot, name, level = step["slot"], self.tower_name(step["slot"]), step["upgrade_to"]
            # count 0: place none, upgrade the towers of this type that earlier plan steps placed
            mine = ([t for t in self.placed if t["slot"] == slot and t.get("step") is not None and t["step"] < i]
                    if step["count"] == 0 else [t for t in self.placed if t.get("step") == i])
            for t in reversed(mine):
                if (t["upgrades"] < level and not t.get("maxed")
                        and time.monotonic() >= t.get("rest_until", 0)):
                    cost = f", next upgrade ${t['next_cost']:,}" if t.get("next_cost") else ""
                    return Goal("upgrade", t, f"plan {i + 1}/{n}: upgrading {name} at spot {t['spot'] + 1} "
                                              f"to level {level} (level {t['upgrades']}{cost})", i)
            if len(mine) >= step["count"]:
                continue
            if self._placement_blocked(slot) is None:
                cost = self.towers.get(slot, {}).get("cost", 0)
                return Goal("place", slot, f"plan {i + 1}/{n}: place {name} {len(mine) + 1} of {step['count']}"
                                           + (f" (${cost:,})" if cost else ""), i)
            first = self.towers.get(slot, {}).get("from_wave", 0)
            if self.reading and 0 < self.reading.wave < first:
                return Goal("hold", None, f"plan {i + 1}/{n}: {name} comes at wave {first}; upgrading meanwhile", i)
        if self.after_plan == "upgrade":
            return Goal("hold", None, "plan done: upgrading only")
        return None

    def _placement_blocked(self, slot):
        """Why tower `slot` can't be placed now (limit, spots, tower cap, its waves), or None."""
        name = self.tower_name(slot)
        limit = self.towers.get(slot, {}).get("limit", 0)
        if limit and self._count(slot) >= limit:
            return f"{name} limit of {limit} reached"
        if self._is_cliff(slot):  # finds its own spot on a cliff; after a failed search, wait a while
            if time.monotonic() < self._cliff_retry.get(slot, 0):
                return f"no cliff found for {name}; looking again in {self._cliff_retry[slot] - time.monotonic():.0f} s"
        elif self.place_target is None and self._free_spot(slot) is None and not self._clusters(slot):
            return f"no free placement spot for {name}"
        r = self.reading
        if r and r.max_towers > 0 and r.towers >= r.max_towers:
            return f"tower limit {r.towers}/{r.max_towers} reached"
        first = self.towers.get(slot, {}).get("from_wave", 0)
        last = self.towers.get(slot, {}).get("until_wave", 10 ** 6)
        if r and r.wave > 0 and not first <= r.wave <= last:
            return f"{name} is only placed in waves {first}-{last if last < 10 ** 6 else 'end'}"
        return None

    def _count(self, slot):
        return sum(t["slot"] == slot for t in self.placed)

    def _saving_for(self):
        """A tower with `save_for: N` that can be placed now while the bot has fewer than N of it:
        until then the bot buys nothing else (no other towers, no upgrades) and saves up for it."""
        for slot, t in self.towers.items():
            want = t.get("save_for", 0)
            if (want and f"PLACE_TOWER_{slot}" in self.actions and self._count(slot) < want
                    and self._placement_blocked(slot) is None):
                return slot
        return None

    def _focus_tower(self):
        """The newest placed tower with `upgrade_to: N` that is still below level N: the bot upgrades
        it (and buys nothing else) before anything new. One strong tower beats several weak ones."""
        for tower in reversed(self.placed):
            level = self.towers.get(tower["slot"], {}).get("upgrade_to", 0)
            if (level and tower["upgrades"] < level and not tower.get("maxed")
                    and time.monotonic() >= tower.get("rest_until", 0)):
                return tower
        return None

    def _focus_note(self, tower):
        level = self.towers[tower["slot"]]["upgrade_to"]
        cost = f", next upgrade ${tower['next_cost']:,}" if tower.get("next_cost") else ""
        return (f"upgrading {self.tower_name(tower['slot'])} at spot {tower['spot'] + 1} to level {level} first "
                f"(level {tower['upgrades']}{cost})")

    def _saving_note(self, slot):
        want = self.towers[slot]["save_for"]
        return f"saving for {self.tower_name(slot)} (has {self._count(slot)} of {want})"

    def _upgradable(self, tower):
        cost = tower.get("next_cost")
        return (not tower.get("maxed") and time.monotonic() >= tower.get("rest_until", 0)
                and (cost is None or self.cash < 0 or cost <= self.cash))

    def vet(self, d):
        """Downgrade impossible or too-frequent actions to WAIT, with a note why."""
        a = d.action
        if a == "WAIT":
            return a, ""
        if self._end_seen:
            return "WAIT", "the match is over (result screen)"
        if self.human_turn:
            return "WAIT", f"wave {self.handover_wave}: the player's turn (recording)"
        if time.monotonic() < self.hold_until:
            return "WAIT", "a new match is loading"
        if time.monotonic() < self.camera.moving_until:
            return "WAIT", "the camera is moving (a boss skill); clicks wait"
        blocked = self._blocked(a)
        if blocked:
            return "WAIT", blocked
        if a.startswith("PLACE_TOWER_"):
            if self.pending:
                return "WAIT", "checking that the last tower was placed"
            cost = self.towers.get(a[-1], {}).get("cost", 0)
            if cost and 0 <= d.cash < cost:
                return "WAIT", f"{a} needs ${cost:,}, cash is ${d.cash:,}"
            if cost and d.affordable.get(a[-1]) is False:
                return "WAIT", f"{a} needs ${cost}, AI says cash is lower"
        group = self._group(a)
        left = self.cooldowns.get(group, 0) - (time.monotonic() - self._last_used.get(group, float("-inf")))
        if left > 0:
            return "WAIT", f"{a} on cooldown ({left:.1f}s)"
        return a, ""

    def execute(self, d):
        action, note = self.vet(d)
        if action == "WAIT":
            return note or "wait"
        if not self.dry_run and self.require_focus and not self.window.is_focused():
            return f"{action} skipped: Roblox is not the focused window"
        region = self.get_region()
        step = None  # set when an action only did its first step (e.g. opened the upgrade panel)
        goal = self._goal()
        try:
            if action.startswith("PLACE_TOWER_"):
                self._place(action[-1], region)
                if goal and goal.kind == "place" and goal.target == action[-1]:
                    self.placed[-1]["step"] = goal.step  # counts towards this build-plan step
            elif action == "UPGRADE_SELECTED":
                step = self._upgrade(region)
            elif action == "SELL":
                self._sell(region)
            elif action == "SKIP_WAVE":
                self._skip(region)
        except Aborted:
            return f"{action} aborted"
        except PlacementBlocked as e:
            self._last_used["PLACE_TOWER"] = time.monotonic()
            skipped = self._step_failed(goal.step) if goal and goal.kind == "place" else None
            return f"{action}: {e}" + (f"; {skipped}" if skipped else "")
        except pdi.FailSafeException:
            self.control.kill()
            return "mouse hit the top-left failsafe corner - killed"
        now = time.monotonic()
        self._last_used[self._group(action)] = now
        if step:
            return step
        self.history.append((now, action))
        if action.startswith("PLACE_TOWER_") and not self.dry_run:
            before = self.reading.towers if self.reading else -1
            self.pending = {"tower": self.placed[-1], "before": before, "t": now}
            self.spent.append((now, self.towers.get(action[-1], {}).get("cost", 0), self.cash))
        return ("[dry-run] " if self.dry_run else "") + action + " done"

    def _put_back(self, key=None):
        """Empty the hand. The cancel key does nothing when no tower is held, so it is pressed either
        way; without one, the tower key puts the tower back (it toggles, so only when one is held)."""
        if self.cancel_key:
            self.input.press(self.cancel_key)
        elif key:
            self.input.press(key)
        self.maybe_holding = False

    def _empty_hand(self):
        """Before clicking the map or reading the shading: a tower still in hand (after a pause in the
        middle of placing) would be dropped by the click, or spoil the free frame."""
        if self.maybe_holding and self.cancel_key:
            self._put_back()
            self.input.wait(0.6)  # let the red shading fade

    def _refused(self, key, region):
        """After the click: did the game keep the tower in hand? Then empty the hand (the cancel key
        is pressed in any case, so a tower kept by mistake can't be dropped by a later click).
        True = refused, False = placed, None = can't tell (no shading was seen)."""
        verdict = self.prober.holding_after(self.input, region)
        if self.cancel_key or verdict:
            self._put_back(key)
            if verdict:
                self.input.wait(0.6)  # the shading fades out; the next look at the screen must not see it
        return verdict

    def _ringless(self):
        """Tower slots the game draws no red ring around (`cluster` towers: Farms)."""
        return {s for s, t in self.towers.items() if t.get("cluster")}

    def _clusters(self, slot):
        """A `cluster` tower (Farm) whose own spots are used up goes next to the first one placed."""
        return bool(self.towers.get(slot, {}).get("cluster")) and any(t["slot"] == slot for t in self.placed)

    def _own_spot(self, slot):
        allowed = self.towers.get(slot, {}).get("spots")
        return self._first_free(allowed, slot) if allowed else None

    def _place_near_own(self, slot, key, region):
        """Farms don't need the path, and every spot near it is worth more with a gun on it: put the next
        one at the clear point nearest the first one (the shading keeps it off the other farms' rings)."""
        self._empty_hand()
        panel_open = not self._close_panel(region)
        name = self.tower_name(slot)
        first = next(t for t in self.placed if t["slot"] == slot)
        self.maybe_holding = True
        found, held, info = self.prober.find_spot(
            self.input, key, region, self._to_screen(self.spots[first["spot"]], region), self._parks(region),
            footprint_px=self.towers[slot].get("footprint_px"), max_dist=self.cluster_px, panel_open=panel_open,
            avoid=self._failed_here(None, region) + self._tower_boxes(region))
        if found is None:
            if self.cancel_key or held:
                self._put_back(key)
            raise PlacementBlocked(f"no clear place near the first {name} ({info})")
        x, y = found
        self.input.click(x, y)
        spot = len(self.spots)
        point = self._from_screen(x, y, region)
        self.spots.append(point)
        self.reserved[spot] = slot
        self.tried[spot] = point
        if self._refused(key, region) and not self.reader_on:
            self.failed_spots[(spot, slot)] = MAX_SPOT_FAILURES
            self.bad_points.append(point)
            raise PlacementBlocked(f"the game refused {name} next to the first one")
        self.placed.append({"slot": slot, "spot": spot, "upgrades": 0})
        self.selected = None

    def _place_on_cliff(self, slot, key, region):
        """Cliff-only tower: the game leaves only cliffs unshaded while it is held, so pick a clear point
        there (near its other towers, else near the middle of the map). No clear point = no cliff."""
        self._empty_hand()
        panel_open = not self._close_panel(region)
        own = [self._to_screen(self.spots[t["spot"]], region) for t in self.placed if t["slot"] == slot]
        self.maybe_holding = True
        found, held, info = self.prober.find_spot(
            self.input, key, region, own[0] if own else None, self._parks(region),
            footprint_px=self.towers[slot].get("footprint_px"), panel_open=panel_open,
            fallback_points=self._cliff_points(slot, region), avoid=self._bad_cliffs(slot, region))
        name = self.tower_name(slot)
        if found is None:
            if self.cancel_key or held or (self.reading is not None
                                           and 0 < self.towers[slot].get("cost", 0) <= self.reading.cash):
                self._put_back(key)
            wait = 300 if info["mode"] == "map" else 60
            self._cliff_retry[slot] = time.monotonic() + wait
            raise PlacementBlocked(f"no cliff for {name} on this map ({info}); looking again in {wait} s")
        x, y = found
        self.input.click(x, y)
        spot = len(self.spots)
        self.spots.append(self._from_screen(x, y, region))
        self.reserved[spot] = slot
        # With the tower counter, a refusal shows as "did not appear" a few seconds later; the look
        # decides only without it.
        if self._refused(key, region) and not self.reader_on:
            self.failed_spots[(spot, slot)] = MAX_SPOT_FAILURES  # never this point again
            raise PlacementBlocked(f"the game refused {name} at that cliff point")
        self.placed.append({"slot": slot, "spot": spot, "upgrades": 0})
        self.selected = None

    def _place(self, slot, region):
        key = self.keys[f"tower_{slot}"]
        if self._is_cliff(slot) and not self.dry_run and self.prober.enabled:
            return self._place_on_cliff(slot, key, region)
        if self.place_target is not None:  # the endgame chose the point (ahead of the boss)
            spot = len(self.spots)
            self.spots.append(tuple(self.place_target))
            self.reserved[spot] = slot
        elif self._clusters(slot) and self._own_spot(slot) is None and not self.dry_run and self.prober.enabled:
            return self._place_near_own(slot, key, region)
        else:
            spot = self._free_spot(slot)
        if spot is None:  # a cliff tower in a dry run: pretend
            spot = len(self.spots)
            self.spots.append((0.5, 0.5))
            self.reserved[spot] = slot
        if self.dry_run or not self.prober.enabled:
            x, y = self._to_screen(self.spots[spot], region)
            self.input.move(x, y)
            self.input.press(key)
            self.input.wait()
            self.input.click(x, y)  # the tower preview follows the cursor; the click drops it
            if self.cancel_key and not self.dry_run:
                self.input.wait(0.2)
                self._put_back()
        else:
            self._empty_hand()
            panel_open = not self._close_panel(region)  # the upgrade panel hides the lower spots
            # Look before clicking: read the game's red no-go shading and pick the nearest clear point.
            points = list(self.prober.candidates(*self.spots[spot], region, panel_open=panel_open))
            self.maybe_holding = True
            found, held, info = self.prober.find_spot(
                self.input, key, region, self._to_screen(self.spots[spot], region), self._parks(region),
                footprint_px=self.towers.get(slot, {}).get("footprint_px"), max_dist=self.prober.max_shift,
                fallback_points=points, panel_open=panel_open,
                avoid=self._failed_here(spot, region) + self._tower_boxes(region, self._ringless()),
                hug=bool(self.towers.get(slot, {}).get("hug_path")))
            if found is None:
                cost = self.towers.get(slot, {}).get("cost", 0)
                if self.cancel_key or held or (self.reading is not None and 0 < cost <= self.reading.cash):
                    self._put_back(key)
                if held:
                    self.failed_spots[(spot, slot)] += 1
                raise PlacementBlocked(f"no clear place near spot {spot + 1} ({info})")
            x, y = found
            self.input.click(x, y)
            self.tried[spot] = self._from_screen(x, y, region)
            refused = self._refused(key, region)
            if refused and not self.reader_on:  # with the tower counter, that decides instead (see observe)
                self.failed_spots[(spot, slot)] += 1
                raise PlacementBlocked(f"the game refused {self.tower_name(slot)} near spot {spot + 1}")
            if not refused:  # remember the position that worked
                self.spots[spot] = self._from_screen(x, y, region)
        self.placed.append({"slot": slot, "spot": spot, "upgrades": 0})
        self.selected = None

    def _select(self, tower, region):
        if self.selected is tower:  # clicking a selected tower again would deselect it
            return
        self._empty_hand()
        if self._under_panel(self.spots[tower["spot"]]):
            self._close_panel(region)  # else the click lands on the open panel, not on this tower
        self.input.click(*self._to_screen(self.spots[tower["spot"]], region))
        self.input.wait(self.input.step * 2)  # let the upgrade panel open
        self.selected = tower
        if self.reader_on:
            self.inspect = {"tower": tower, "t": time.monotonic()}

    def _upgrade(self, region):
        """Keep upgrading the selected tower while that works, otherwise the next one in turn.

        With the number reader the bot first opens the tower's panel and reads the price, and
        only presses the upgrade key when the cash covers it. A fully upgraded tower is skipped.
        """
        goal = self._goal()
        tower = self.selected if self.selected is not None and self._upgradable(self.selected) else None
        if goal is not None and goal.kind == "upgrade":
            tower = goal.target if self._upgradable(goal.target) else None
        elif tower is None:
            n = len(self.placed)
            for k in range(n):
                candidate = self.placed[(self._next_upgrade + k) % n]
                if self._upgradable(candidate):
                    tower, self._next_upgrade = candidate, self._next_upgrade + k + 1
                    break
        if tower is None:
            return "nothing to upgrade"
        name, spot = self.tower_name(tower["slot"]), tower["spot"] + 1
        if self.selected is not tower:
            self._select(tower, region)
            if self.reader_on:
                return f"opened {name}'s upgrade panel (spot {spot})"
        elif self.reader_on and tower.get("next_cost") is None:
            return f"reading {name}'s upgrade price"
        self.input.press(self.keys["upgrade"])
        tower["upgrades"] += 1
        if tower.get("next_cost"):
            self.spent.append((time.monotonic(), tower["next_cost"], self.cash))
        # With the tower's prices known, the next one is known too: the next tick can press again without
        # waiting a second for the reader (it still checks the panel and corrects the level if needed).
        costs = self.towers.get(tower["slot"], {}).get("upgrade_costs")
        tower["next_cost"] = costs[tower["upgrades"]] if costs and tower["upgrades"] < len(costs) else None
        if costs and tower["upgrades"] >= len(costs):
            tower["maxed"] = True
        if self.reader_on:
            self.inspect = {"tower": tower, "t": time.monotonic()}
        return None

    def _sell(self, region):
        goal = self._goal()
        if goal is not None and goal.kind == "sell":
            tower = goal.target
        else:
            tower = self.selected if self.selected is not None else self.placed[-1]
        self._select(tower, region)
        self.input.press(self.keys["sell"])
        self._forget(tower)
        self.inspect = None

    def _skip(self, region):
        if self.skip_button:
            self.input.click(*self._to_screen(self.skip_button, region, ui=True))
        else:
            self.input.press(self.keys["skip_wave"])
