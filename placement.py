"""Find a position where the game accepts a tower, by looking at the screen.

While a tower is held for placing, TDS shades every place it can't go in red: the path, scenery,
and a red ring around each tower. So the bot takes one frame before pressing the tower key and
two after, with the cursor at opposite edges of the screen. What changed in both is the no-go
map (a tower's ring is filled in); the held tower's own range circle and its Place/Rotate menu
follow the cursor, so they show up in only one of the two. It then picks the point nearest the
wanted spot where the tower's whole footprint is clear, and clicks there. A cliff-only tower
(Ranger) sees only cliffs left unshaded, so the same map shows the cliffs.

If no red shading shows up at all, it falls back to moving the preview around the spot and
comparing each patch with a reference ("ok" / "blocked" / "hidden").

Images of what it saw are saved to probe_debug/ (the first few per run) for checking.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import mss
import numpy as np

ROOT = Path(__file__).resolve().parent
DEBUG_DIR = ROOT / "probe_debug"
REFERENCE_WIDTH = 1920  # pixel settings in config are for a 1920-wide game window and scale with it
# TDS screen UI never to click through (x, y, w, h as window fractions): Roblox menu buttons, the wave /
# base health / time bar, the players list with coins and settings, the bottom-left buttons.
SCREEN_UI = [(0, 0, 0.12, 0.06), (0.32, 0, 0.38, 0.095), (0.8, 0, 0.2, 0.14), (0, 0.93, 0.07, 0.07)]


def ring_offsets(radii):
    """(0, 0) first, then rings of points, nearest first; more points on the larger rings."""
    yield 0, 0
    for r in radii:
        if r <= 0:
            continue
        n = max(8, round(2 * math.pi * r / 28))
        for k in range(n):
            a = 2 * math.pi * k / n
            yield round(r * math.cos(a)), round(r * math.sin(a))


def sell_button_visible(crop):
    """Is a tower's upgrade panel open? Look for its Sell button: a wide, solidly filled red bar.

    Searched anywhere in the crop (the upgrade panel area), so panels laid out a little
    differently still count, while red scenery (cars, tractors) is the wrong shape.
    """
    if crop is None or crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 120)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(red)
    height, width = crop.shape[:2]
    for x, y, w, h, area in stats[1:]:
        if w >= 0.15 * width and 0.02 * height <= h <= 0.2 * height and w >= 3 * h and area >= 0.6 * w * h:
            return True
    return False


def upgrade_bars_full(box):
    """In the next-upgrade box: are all the level bars lit bright green? That is "Fully Upgraded!".
    The price box also turns green when an upgrade is affordable, and then the vision model tends to
    call it fully upgraded; the bars tell the two apart (affordable: some or none lit)."""
    if box is None or box.size == 0:
        return False
    bh, bw = box.shape[:2]
    strip = box[int(0.78 * bh):int(0.98 * bh), int(0.05 * bw):int(0.97 * bw)].astype(np.int16)
    b, g, r = strip[..., 0], strip[..., 1], strip[..., 2]
    lime = (g >= 170) & (g - r >= 60) & (g - b >= 60)
    return float(lime.mean()) >= 0.15 and float(lime.any(axis=0).mean()) >= 0.75


def red_tint(img):
    """Pixels shaded TDS's no-placement red (pinkish red, not the sand or grass underneath)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 12) | (h >= 170)) & (s >= 50) & (v >= 110)


def turned_red(free, held, threshold=15, changed_sum=45):
    """Pixels the red shading covers once a tower is held: those that got redder, or changed clearly.

    Only no-go areas get shaded; valid ground stays as it was (apart from the preview itself and
    moving enemies, which are small). Checking for any clear change, not just "redder", also works
    on ground that is red or orange already, where the shading barely shifts the hue.
    """
    f, h = free.astype(np.int16), held.astype(np.int16)
    d = h - f  # BGR
    redder = (d[..., 2] - (d[..., 1] + d[..., 0]) / 2) > threshold
    return redder | (np.abs(d).sum(axis=2) > changed_sum)


def no_go_map(free, held, ring_fill_px=8000, speck_px=12):
    """Where a tower can't go: what turned red once it was held (free = before the key, held = one
    frame after, or several taken with the cursor in different places: only what changed in all
    of them counts, so whatever follows the cursor drops out).

    Small clear areas fully enclosed by red (inside another tower's ring) count as no-go too;
    grass islands between path segments are much bigger and stay clear. Returns (mask, share of
    the frame that turned red); a tiny share means the shading wasn't seen.
    """
    helds = held if isinstance(held, (list, tuple)) else [held]
    new = np.logical_and.reduce([turned_red(free, h) if free is not None else red_tint(h) for h in helds])
    m = new.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)  # drop specks (moving enemies)
    specks = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < speck_px]
    if specks:
        m[np.isin(labels, specks)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # gaps in tower rings
    share = float((m > 0).mean())
    clear = (m == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clear, connectivity=4)
    small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < ring_fill_px]
    if small:
        m[np.isin(labels, small)] = 255
    return m > 0, share


def pick_point(no_go, target, footprint_px, max_dist=None, keep_out=()):
    """The point nearest `target` (x, y in mask pixels) whose whole footprint is clear, or None."""
    clear = (~no_go).astype(np.uint8)
    for x, y, w, h in keep_out:  # HUD and panels, as pixel boxes
        clear[max(0, y):y + h, max(0, x):x + w] = 0
    r = max(1, int(footprint_px))
    core = cv2.erode(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    ys, xs = np.nonzero(core)
    if xs.size == 0:
        return None
    d = (xs - target[0]) ** 2 + (ys - target[1]) ** 2
    i = int(d.argmin())
    if max_dist is not None and d[i] > max_dist ** 2:
        return None
    return int(xs[i]), int(ys[i])


def nearest_path_point(no_go, target, max_dist=None):
    """The point of the path (the largest no-go area) nearest `target`, or `target` if none is near."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(no_go.astype(np.uint8), connectivity=8)
    if n < 2:
        return target
    path = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == path)
    d = (xs - target[0]) ** 2 + (ys - target[1]) ** 2
    i = int(d.argmin())
    if max_dist is not None and d[i] > max_dist ** 2:
        return target
    return float(xs[i]), float(ys[i])


def shrink(img, k):
    return cv2.resize(img, (img.shape[1] // k, img.shape[0] // k), interpolation=cv2.INTER_AREA)


def ghost_verdict(before, after, red_ratio, min_changed):
    """Compare a patch without the ghost (before) and with it (after)."""
    changed = cv2.absdiff(before, after).max(axis=2) > 35
    n = int(changed.sum())
    stats = {"changed": round(n / changed.size, 3), "red": 0.0}
    if n < min_changed * changed.size:
        return "hidden", stats
    hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h <= 10) | (h >= 170)) & (s >= 110) & (v >= 80) & changed
    stats["red"] = round(float(red.sum()) / n, 3)
    return ("blocked" if stats["red"] >= red_ratio else "ok"), stats


class Prober:
    def __init__(self, cfg):
        p = cfg.get("placement", {})
        self.enabled = p.get("probe", True)
        self.radii = p.get("search_px", [0, 28, 56, 84])
        self.patch = p.get("patch_px", 60)
        self.settle = p.get("settle_s", 0.1)
        self.budget = p.get("max_probe_s", 3.0)
        self.red_ratio = p.get("red_ratio", 0.2)
        self.min_changed = p.get("min_changed", 0.015)
        self.debug_left = p.get("debug_images", 12)
        self.overlay_debug_left = p.get("debug_images", 12)
        self.footprint = p.get("footprint_px", 28)       # tower's clear radius, at 1920 wide
        self.max_shift = p.get("max_shift_px", 150)      # how far from its spot a ground tower may move
        self.overlay_wait = p.get("overlay_wait_s", 0.4)  # time for the red shading to appear
        self.min_overlay = p.get("min_overlay", 0.005)    # share of the frame that must turn red to trust it
        self.park_wait = p.get("park_wait_s", 0.15)       # after moving the held tower to the second park point
        self.confirm_wait = p.get("confirm_wait_s", 0.6)  # after a click, before checking the tower left the hand
        self._free = self._held = self._shade = self._park = None  # last find_spot, for holding_after
        self._last_view = None  # (held frame, no-go map, target, chosen point, footprint) of the last placement
        self.no_go = [cfg.get("rois", {})[k] for k in ("cash", "towers_count", "wave", "hotbar")
                      if k in cfg.get("rois", {})] + [list(b) for b in SCREEN_UI]
        self.panel = cfg.get("rois", {}).get("upgrade_panel")
        self._sct = None  # created in the thread that uses it (mss handles are per thread)

    def grab(self, box):
        if self._sct is None:
            self._sct = getattr(mss, "MSS", mss.mss)()
        return cv2.cvtColor(np.asarray(self._sct.grab(box)), cv2.COLOR_BGRA2BGR)

    def candidates(self, fx, fy, region, panel_open):
        """Screen points around a spot (fractions), skipping the HUD and, while it is open, the upgrade panel."""
        scale = region["width"] / REFERENCE_WIDTH
        blocked = self.no_go + ([self.panel] if panel_open and self.panel else [])
        for dx, dy in ring_offsets([r * scale for r in self.radii]):
            gx, gy = fx + dx / region["width"], fy + dy / region["height"]
            if not (0.02 < gx < 0.98 and 0.02 < gy < 0.98):
                continue
            if any(x <= gx <= x + w and y <= gy <= y + h for x, y, w, h in blocked):
                continue
            yield round(region["left"] + gx * region["width"]), round(region["top"] + gy * region["height"])

    def patch_box(self, x, y, region):
        r = round(self.patch * region["width"] / REFERENCE_WIDTH)
        left, top = max(region["left"], x - r), max(region["top"], y - r)
        right = min(region["left"] + region["width"], x + r)
        bottom = min(region["top"] + region["height"], y + r)
        return {"left": left, "top": top, "width": right - left, "height": bottom - top}

    def find_spot(self, inp, key, region, target, parks, footprint_px=None, max_dist=None, fallback_points=(),
                  panel_open=False, avoid=(), hug=False):
        """Hold `key` and pick a clear point from the game's red no-go shading.

        `target` is the wanted screen point (None = near the middle of the map); `max_dist` limits
        how far from it the tower may move (None = anywhere, for cliff towers). `parks` are screen
        points far apart where the cursor waits while the frames are taken. Returns (screen point
        or None, whether a tower is held, info). If no shading shows up, sweeps the preview over
        `fallback_points` the old way. `hug`: a melee tower (Warden) goes right next to the path, at the
        clear point nearest the piece of path closest to `target`.
        """
        k = 2  # work at half resolution
        scale = region["width"] / REFERENCE_WIDTH
        inp.move(*parks[0])
        free = shrink(self.grab(region), k)
        inp.press(key)
        inp.wait(self.overlay_wait)
        helds = [shrink(self.grab(region), k)]
        for park in parks[1:]:
            inp.move(*park)
            inp.wait(self.park_wait)
            helds.append(shrink(self.grab(region), k))
        self._park = parks[-1]
        self._free, self._held = free.astype(np.int16), helds[-1].astype(np.int16)
        self._shade = np.logical_and.reduce([turned_red(free, h) for h in helds])
        no_go, share = no_go_map(free, helds, ring_fill_px=8000 * scale * scale / (k * k))
        if share < self.min_overlay:  # no red shading seen: sweep the preview instead
            self._shade = None
            saw, log = False, []
            reference = self.grab(region)
            for x, y in fallback_points:
                if any(bx <= x <= bx + bw and by <= y <= by + bh for bx, by, bw, bh in avoid):
                    continue  # this tower already failed there
                verdict, stats = self.test(inp, x, y, region, reference)
                saw |= verdict != "hidden"
                log.append((x, y, verdict, stats))
                if verdict == "ok":
                    return (x, y), True, {"mode": "sweep", "tried": len(log)}
            return None, saw, {"mode": "sweep", "tried": len(log)}
        h, w = no_go.shape
        boxes = self.no_go + ([self.panel] if panel_open and self.panel else [])
        margin = [(0, 0, 0.02, 1), (0.98, 0, 0.02, 1), (0, 0, 1, 0.04), (0, 0.97, 1, 0.03)]
        keep_out = [(int(x * w), int(y * h), int(bw * w) + 1, int(bh * h) + 1) for x, y, bw, bh in boxes + margin]
        keep_out += [((x - region["left"]) // k, (y - region["top"]) // k, bw // k + 1, bh // k + 1)
                     for x, y, bw, bh in avoid]  # places where this tower already failed
        tx, ty = ((target[0] - region["left"]) / k, (target[1] - region["top"]) / k) if target else (w / 2, h / 2)
        fp = (footprint_px or self.footprint) * scale / k
        limit = max_dist * scale / k if max_dist else None
        if hug:
            tx, ty = nearest_path_point(no_go, (tx, ty), limit)
        point = pick_point(no_go, (tx, ty), fp, limit, keep_out)
        if point is None and avoid:  # the places that failed before cover everything near: try anyway
            point = pick_point(no_go, (tx, ty), fp, limit, keep_out[:len(boxes) + len(margin)])
        self._save_overlay_debug(helds[-1], no_go, (tx, ty), point, fp)
        self._last_view = (helds[-1], no_go, (tx, ty), point, fp)
        info = {"mode": "map", "red": round(share, 3)}
        if point is None:
            return None, True, info
        return (region["left"] + point[0] * k, region["top"] + point[1] * k), True, info

    def holding_after(self, inp, region):
        """After the click: is the tower still in hand (the game refused the spot)?

        Moves the cursor back to the last park point and waits for the shading to fade, then asks
        whether the no-go areas look like the held frame or the free one. Looking right at the click
        point too soon is fooled: the new tower's range shows while the cursor is on it, and the
        shading takes a moment to fade. True / False, or None when there was no shading to compare.
        """
        if self._shade is None or int(self._shade.sum()) < 50:
            return None
        inp.move(*self._park)
        inp.wait(self.confirm_wait)
        now = shrink(self.grab(region), 2).astype(np.int16)
        d_held = float(np.abs(now - self._held).sum(axis=2)[self._shade].mean())
        d_free = float(np.abs(now - self._free).sum(axis=2)[self._shade].mean())
        return d_held < d_free

    def save_failure(self, label):
        """The game didn't take the last placement: save what the bot saw (probe_debug/fail_*.png)."""
        if self._last_view is not None:
            self._save_overlay_debug(*self._last_view, name=f"fail_{time.strftime('%H%M%S')}_{label}")
            self._last_view = None
            for old in sorted(DEBUG_DIR.glob("fail_*.png"), key=lambda f: f.stat().st_mtime)[:-40]:
                old.unlink(missing_ok=True)  # long unattended runs: keep the newest 40

    def _save_overlay_debug(self, held, no_go, target, point, fp, name=None):
        if name is None:
            if self.overlay_debug_left <= 0:
                return
            self.overlay_debug_left -= 1
            name = f"map_{time.strftime('%H%M%S')}_{self.overlay_debug_left:02d}"
        DEBUG_DIR.mkdir(exist_ok=True)
        view = held.copy()
        tint = view.copy()
        tint[no_go] = (0, 0, 200)
        view = cv2.addWeighted(tint, 0.45, view, 0.55, 0)
        cv2.drawMarker(view, (int(target[0]), int(target[1])), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        if point:
            cv2.circle(view, point, max(3, int(fp)), (0, 255, 0), 2)
        label = "chosen: green circle" if point else "no clear place"
        cv2.putText(view, f"red = no-go, cross = wanted spot, {label}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(DEBUG_DIR / f"{name}.png"), view)

    def begin(self, inp, key, park, region):
        """Press `key` with the cursor parked far away; returns the reference frame (preview not near the targets)."""
        inp.move(*park)
        inp.press(key)
        inp.wait(0.15)
        return self.grab(region)

    def rebase(self, inp, park, region):
        """A fresh reference frame with the preview (already held) parked again; the scene keeps moving."""
        inp.move(*park)
        inp.wait(self.settle)
        return self.grab(region)

    def test(self, inp, x, y, region, reference):
        """Move the preview to (x, y) and judge it: ("ok" | "blocked" | "hidden", stats)."""
        inp.move(x, y)
        inp.wait(self.settle)
        box = self.patch_box(x, y, region)
        after = self.grab(box)
        ox, oy = box["left"] - region["left"], box["top"] - region["top"]
        before = reference[oy:oy + box["height"], ox:ox + box["width"]]
        verdict, stats = ghost_verdict(before, after, self.red_ratio, self.min_changed)
        self._save_debug(before, after, verdict, stats)
        return verdict, stats

    def find(self, inp, key, park, points, region, budget=None):
        """Press `key`, then try `points` until one is accepted; returns (point or None, saw_ghost, log)."""
        reference = self.begin(inp, key, park, region)
        saw_ghost, log, t0 = False, [], time.monotonic()
        for x, y in points:
            if time.monotonic() - t0 > (budget or self.budget):
                break
            verdict, stats = self.test(inp, x, y, region, reference)
            saw_ghost |= verdict != "hidden"
            log.append((x, y, verdict, stats))
            if verdict == "ok":
                return (x, y), True, log
        return None, saw_ghost, log

    def _save_debug(self, before, after, verdict, stats):
        if self.debug_left <= 0:
            return
        self.debug_left -= 1
        DEBUG_DIR.mkdir(exist_ok=True)
        side = np.hstack([before, after])
        label = f"{verdict}  changed={stats['changed']:.3f}  red={stats['red']:.3f}"
        bar = np.zeros((22, side.shape[1], 3), np.uint8)
        cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(DEBUG_DIR / f"{time.strftime('%H%M%S')}_{self.debug_left:02d}_{verdict}.png"),
                    np.vstack([bar, side]))
