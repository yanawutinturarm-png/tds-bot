"""The end of a Molten match, played the way the player won it (recordings/20261018-232625-wave34).

Wave 34 (`endgame.from_wave`):
  1. Sell the Farms (their income no longer matters) and towers far from the path (out of range).
  2. Bring the Crook Bosses, then the other towers, to their top level.
  3. Keep the rest of the cash for the boss.
Wave 35, the Molten Warlord (boss bar up):
  4. Crook Bosses target "Strongest": two clicks on the panel's target arrow (First -> Last -> Strongest).
     The Warlord has by far the most health.
  5. Leapfrog Wardens. The Warlord walks the path slowly; Wardens are melee. Sell the Wardens it has walked
     past and place new ones on the path just ahead of it, each straight to level 4 (stun immune). Its
     $2,500 per 1% of health pays for the moves.

Where the boss is: `BossTracker`, from the screen (the largest yellow-orange blob on the path near where
it should be by now) and, between sightings, its walking speed.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


class PathLine:
    """The enemy path as a line of map points (fractions), from the spawn to the base."""

    def __init__(self, points):
        self.p = np.float32(points)
        seg = np.hypot(np.diff(self.p[:, 0]) * 16, np.diff(self.p[:, 1]) * 9)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def locate(self, x, y):
        """(progress 0..1 of the nearest point on the path, distance from it in pixels at 1920x1080)."""
        a, b = self.p[:-1], self.p[1:]
        ab = (b - a) * (16, 9)
        ap = (np.float32([x, y]) - a) * (16, 9)
        t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        off = ap - ab * t[:, None]
        d = np.hypot(off[:, 0], off[:, 1])
        i = int(d.argmin())
        return float((self.cum[i] + t[i] * (self.cum[i + 1] - self.cum[i])) / self.length), float(d[i]) * 120

    def point(self, progress):
        """The map point at this progress along the path."""
        s = min(max(progress, 0.0), 1.0) * self.length
        i = min(max(int(np.searchsorted(self.cum, s, side="right")) - 1, 0), len(self.p) - 2)
        f = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9)
        x, y = self.p[i] + (self.p[i + 1] - self.p[i]) * f
        return float(x), float(y)


class BossTracker:
    """How far the boss has walked (0 = spawn, 1 = base). The Molten Warlord is a big grey robot with bright
    yellow-orange parts: the largest yellow-orange blob on the path, near where it should be by now, is
    taken as the boss. Yellow things that were there before it came are left out. Between sightings it
    walks on at `speed` (path per second, measured in the winning recording); it never walks back."""

    W, H = 960, 540

    def __init__(self, path, speed=0.007, band_px=70, min_area=250):
        self.path, self.speed, self.band_px, self.min_area = path, speed, band_px, min_area
        self.reset()

    def reset(self):
        self.started = None     # when the boss appeared
        self.seen_p = 0.0       # progress at the last sighting (or the spawn)
        self.seen_t = None
        self.sightings = 0
        self._static = np.zeros((self.H, self.W), np.float32)
        self._static_n = 0

    def _band(self, to_screen):
        band = np.zeros((self.H, self.W), np.uint8)
        pts = np.int32([[x * self.W, y * self.H] for x, y in (to_screen(*p) for p in self.path.p)])
        cv2.polylines(band, [pts], False, 255, self.band_px)
        return band > 0

    def _yellow(self, frame, band):
        small = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = (h >= 14) & (h <= 30) & (s >= 150) & (v >= 170) & band
        m[:int(0.2 * self.H)] = False    # wave, base health, boss bar
        m[int(0.88 * self.H):] = False   # hotbar, cash
        return m

    def learn_static(self, frame, to_screen=lambda x, y: (x, y)):
        """Before the boss: note what is yellow on the path anyway (a tower's paint, a crate)."""
        self._static += self._yellow(frame, self._band(to_screen))
        self._static_n += 1

    def start(self, now):
        self.started, self.seen_p, self.seen_t = now, 0.0, now

    def estimate(self, now):
        if self.started is None:
            return None
        return min(1.0, self.seen_p + min(self.speed * (now - self.seen_t), 0.15))

    def update(self, frame, now, to_screen=lambda x, y: (x, y), to_map=lambda x, y: (x, y), hide=()):
        """Look for the boss in this frame; returns the progress estimate. `hide`: screen boxes (fractions
        x, y, w, h) to leave out, e.g. an open upgrade panel."""
        if self.started is None:
            return None
        expected = self.estimate(now)
        m = self._yellow(frame, self._band(to_screen))
        if self._static_n >= 5:
            m &= ~cv2.dilate((self._static / self._static_n >= 0.4).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        for x, y, w, h in hide:
            m[int(y * self.H):int((y + h) * self.H), int(x * self.W):int((x + w) * self.W)] = False
        m = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(m)
        best = None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            q, _ = self.path.locate(*to_map(cents[i][0] / self.W, cents[i][1] / self.H))
            if self.seen_p - 0.03 <= q <= expected + 0.08 and (best is None or area > best[0]):
                best = (area, q)
        if best is not None:
            self.seen_p, self.seen_t = max(self.seen_p, best[1]), now
            self.sightings += 1
        return self.estimate(now)


class Endgame:
    """Runs the end of the match instead of the build plan (see the module notes). `tick` is called by the
    bot every round; it returns None while the endgame hasn't started, else a note of what it did."""

    def __init__(self, cfg, executor, latest_frame=None):
        c = cfg.get("endgame", {})
        self.on = bool(c.get("enabled", True)) and len(cfg.get("path") or []) >= 2
        self.ex = executor
        self.latest_frame = latest_frame or (lambda: None)
        self.from_wave = c.get("from_wave", 34)
        self.boss_wave = c.get("boss_wave", 35)
        self.sell_names = set(c.get("sell", ["Farm"]))
        self.sell_far_px = c.get("sell_far_px", 180)
        self.keep_names = set(c.get("keep", ["Warden", "Crook Boss", "Commander"]))
        self.max_out = c.get("max_out", ["Crook Boss", "Warden", "Commander"])
        self.target = c.get("target", {"Crook Boss": 2})  # tower -> clicks on the target arrow
        # The boss wave: "lose" sells every tower so the match ends lost (a win sends the game to the lobby,
        # where the bot can't go on; a loss at 35 still pays coins and XP and offers Restart Match);
        # "win" plays it (Strongest, leapfrog Wardens); "idle" just stops.
        self.final = c.get("final_wave", "lose")
        self.arrow = tuple(c.get("target_arrow", [0.454, 0.755]))
        lf = c.get("leapfrog", {})
        self.lf_name = lf.get("tower", "Warden")
        self.behind = lf.get("behind", 0.05)
        self.ahead = tuple(lf.get("ahead", [0.08, 0.2]))
        self.keep_cash = lf.get("keep_cash", 2500)
        self.start_after = lf.get("start_after", 0.15)  # the player began once the boss was this far along
        self.path = PathLine(cfg["path"]) if self.on else None
        self.boss = BossTracker(self.path, c.get("boss_speed", 0.007)) if self.on else None
        self.reset()

    def reset(self):
        self.active = False
        self.giving_up = False
        self.progress = None
        self._targeted = set()   # id() of towers already set to Strongest
        self._bad = []           # (progress, until): leapfrog targets where no place was found
        if self.boss:
            self.boss.reset()

    # ---- helpers
    def _name(self, t):
        return self.ex.tower_name(t["slot"])

    def _where(self, t):
        """(progress along the path, distance from it in px) of a placed tower."""
        spots = self.ex.spots
        return self.path.locate(*spots[t["spot"]]) if t["spot"] < len(spots) else (0.0, 0.0)

    def _top(self, t):
        tw = self.ex.towers.get(t["slot"], {})
        return tw.get("max_level") or len(tw.get("upgrade_costs") or []) or 0

    def _below_top(self, t):
        return t["upgrades"] < self._top(t) and not t.get("maxed")

    def _full_cost(self, slot):
        tw = self.ex.towers.get(slot, {})
        return tw.get("cost", 0) + sum(tw.get("upgrade_costs") or [])

    def _slot_of(self, name):
        return next((s for s, t in self.ex.towers.items() if t.get("name") == name), None)

    # ---- the round
    def tick(self, frame, reading, region):
        ex = self.ex
        if not self.on or getattr(ex, "human_turn", False):
            return None
        now = time.monotonic()
        if not self.active:
            if ex._end_seen or now < ex.hold_until:
                return None
            if ex.wave < self.from_wave:  # the confirmed wave (never one misread number, see _confirm_wave)
                return None
            self.active = True
        if ex.wave >= self.boss_wave and self.final in ("lose", "idle"):
            return self._give_up(reading, region)
        cam = ex.camera
        # once the boss is out it stays the boss fight, even through a misread wave number
        boss_up = self.boss.started is not None or (ex.wave >= self.boss_wave and _boss_bar(frame))
        if boss_up and self.boss.started is None:
            self.boss.start(now)
        if self.boss.started is None:
            if frame is not None:
                self.boss.learn_static(frame, cam.to_screen)
        elif frame is not None:
            hide = [ex.prober.panel] if ex.selected is not None and ex.prober.panel else []
            self.progress = self.boss.update(frame, now, cam.to_screen, cam.to_map, hide)
        if ex._end_seen:
            return "endgame: the match is over"
        if now < ex.hold_until or now < cam.moving_until:
            return "endgame: waiting (the view is moving)"
        task = self._next_task(boss_up)
        if task is None:
            where = f"; boss at {self.progress:.2f} of the path" if self.progress is not None else ""
            return f"endgame: holding the cash for the boss{where}" if not boss_up else f"endgame: nothing to do{where}"
        return self._do(task, reading, region)

    def _next_task(self, boss_up):
        ex = self.ex
        towers = list(ex.placed)
        cash = ex.cash
        # 1. sell the Farms and towers out of the path's reach
        for t in towers:
            name = self._name(t)
            far = self._where(t)[1]
            if name in self.sell_names or (name not in self.keep_names and far > self.sell_far_px):
                why = "no more income needed" if name in self.sell_names else f"{far:.0f} px from the path"
                return "sell", t, f"endgame: sell {name} at spot {t['spot'] + 1} ({why})"
        if boss_up:
            # 2. Crook Bosses (etc.) target the strongest enemy: the Warlord
            for t in towers:
                clicks = self.target.get(self._name(t), 0)
                if clicks and id(t) not in self._targeted:
                    return "target", (t, clicks), f"endgame: {self._name(t)} at spot {t['spot'] + 1} targets Strongest"
            # 3. a leapfrog tower just placed: straight to its top level before any other move
            fresh = [t for t in towers if t.get("leapfrog") and self._below_top(t)]
            for t in fresh:
                if ex._upgradable(t):
                    return "upgrade", t, f"endgame: new {self._name(t)} at spot {t['spot'] + 1} to level {self._top(t)}"
            if fresh:
                return "wait", None, f"endgame: saving for the new {self._name(fresh[0])}'s next level"
            lf = self._leapfrog(towers, cash)
            if lf:
                return lf
        # 4. everything to its top level, Crook Bosses first
        for name in self.max_out:
            for t in towers:
                if self._name(t) == name and self._below_top(t) and ex._upgradable(t):
                    return "upgrade", t, f"endgame: {name} at spot {t['spot'] + 1} to level {self._top(t)}"
        return None

    def _leapfrog(self, towers, cash):
        p = self.progress
        slot = self._slot_of(self.lf_name)
        if p is None or slot is None or p < self.start_after:
            return None
        mine = [(self._where(t)[0], t) for t in towers if t["slot"] == slot]
        full = self._full_cost(slot)
        limit = self.ex.towers.get(slot, {}).get("limit", 0) or 10 ** 6
        if len(mine) < limit and 0 <= cash and cash >= self.ex.towers[slot].get("cost", 0) + self.keep_cash and p < 0.95:
            target = self._ahead_target(p, [q for q, _ in mine])
            if target is not None:
                return "place", (slot, target), f"endgame: {self.lf_name} ahead of the boss (boss {p:.2f}, at {target:.2f})"
        behind = sorted(((q, t) for q, t in mine if q < p - self.behind), key=lambda qt: qt[0])
        refund = 0.35 * full  # what selling a top-level one gives back, about
        if behind and len(mine) >= limit and cash + refund >= full + self.keep_cash:
            q, t = behind[0]
            return "sell", t, f"endgame: sell the {self.lf_name} the boss walked past (it {q:.2f}, boss {p:.2f})"
        return None

    def _ahead_target(self, p, have):
        now = time.monotonic()
        self._bad = [(q, u) for q, u in self._bad if u > now]
        best = None
        for k in range(int(round((self.ahead[1] - self.ahead[0]) / 0.01)) + 1):
            q = round(p + self.ahead[0] + 0.01 * k, 3)
            if q > 0.97 or any(abs(q - b) < 0.02 for b, _ in self._bad):
                continue
            gap = min((abs(q - h) for h in have), default=1.0)
            if best is None or gap > best[0] + 1e-6:
                best = (gap, q)
        return best[1] if best and best[0] >= 0.02 else None

    def _do(self, task, reading, region):
        from brain import Decision
        from controls import Aborted, Goal
        ex = self.ex
        kind, target, why = task
        if kind == "wait":
            return why
        if kind == "target":
            return self._set_target(*target, why, region)
        decision = Decision(action="", cash=ex.cash, wave=ex.wave, reason=why, source="endgame")
        if kind == "sell":
            decision.action, ex.forced_goal = "SELL", Goal("sell", target, why)
        elif kind == "upgrade":
            decision.action, ex.forced_goal = "UPGRADE_SELECTED", Goal("upgrade", target, why)
        else:  # place
            slot, q = target
            decision.action, ex.forced_goal = f"PLACE_TOWER_{slot}", Goal("place", slot, why)
            ex.place_target = self.path.point(q)
        n_before = len(ex.placed)
        try:
            result = ex.execute(decision)
        except Aborted:
            result = f"{decision.action} aborted"
        finally:
            ex.forced_goal, ex.place_target = None, None
        if kind == "place":
            if len(ex.placed) > n_before:
                ex.placed[-1]["leapfrog"] = True
            elif "no clear place" in result or "refused" in result:
                self._bad.append((target[1], time.monotonic() + 15))
        return f"{why}: {result}"

    def _give_up(self, reading, region):
        """The boss wave, lost on purpose: no upgrades, no abilities, and ("lose") every tower sold, the
        damage dealers first, so the base falls quickly and the Restart Match button comes up."""
        ex = self.ex
        if not self.giving_up:
            self.giving_up = True
            ex.cooldowns = {**ex.cooldowns, "SELL": min(ex.cooldowns.get("SELL", 1.0), 0.4)}
        if ex._end_seen:
            return "endgame: the match is over"
        if self.final == "idle":
            return f"endgame: wave {ex.wave} not played (a win would end in the lobby)"
        order = {"Warden": 0, "Crook Boss": 1, "Commander": 2}
        towers = sorted(ex.placed, key=lambda t: order.get(self._name(t), 3))
        if not towers:
            return f"endgame: wave {ex.wave}: everything sold, waiting for the match to end"
        t = towers[0]
        why = f"endgame: wave {ex.wave} is given up (a win would end in the lobby): sell {self._name(t)} at spot {t['spot'] + 1}"
        return self._do(("sell", t, why), reading, region)

    def _set_target(self, tower, clicks, why, region):
        """Open the tower's panel and click the target arrow `clicks` times."""
        from controls import Aborted
        ex = self.ex
        try:
            ex._select(tower, region)
            ex.input.wait(0.3)
            if not ex.dry_run and not ex._panel_visible(region):
                ex.selected = None  # the panel didn't open: try again next round
                return f"{why}: the panel didn't open; trying again"
            for _ in range(clicks):
                ex.input.click(*ex._to_screen(self.arrow, region, ui=True))
                ex.input.wait(0.35)
        except Aborted:
            return f"{why}: aborted"
        self._targeted.add(id(tower))
        return f"{why}: done" + (" [dry-run]" if ex.dry_run else "")


def _boss_bar(frame):
    from controls import boss_bar_visible
    return frame is not None and boss_bar_visible(frame)
