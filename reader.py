"""The bot's eyes for numbers: reads cash, wave and the tower counter exactly.

A decision model like Jev can't output numbers, so a small local vision LLM (gemma3:4b
through Ollama, free) reads them in its own thread. Decisions never wait for it; they use
the latest reading.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from brain import BrainError, _crop, _jpeg_b64, _panel, _post_json, _to_int
from placement import sell_button_visible, upgrade_bars_full

FIELDS = ("cash", "wave", "total_waves", "towers", "max_towers", "upgrade_cost")
# Bare numbers instead of a JSON object with keys: far fewer tokens to generate, so a read is ~3x faster.
PROMPT = ("These are labeled crops of the Roblox Tower Defense Simulator UI. CASH is the player's money "
          "(ignore the $ sign and commas). WAVE reads 'current / total'. TOWERS reads 'placed / maximum'. "
          "UPGRADE is the selected tower's next-upgrade box, showing a price such as $1,250, or 'Fully Upgraded!'. "
          "Reply with a JSON array of six integers: [cash, current wave, total waves, towers placed, maximum towers, "
          "upgrade price]. For the upgrade price use 0 if it says Fully Upgraded or Upgrade Locked, and -1 if the "
          "UPGRADE crop shows no upgrade box, only the game world. Use -1 for anything else you cannot read.")
# The players list's cash, a second opinion on a sudden rise (the counter at the bottom sits on a busy
# background). A separate small request: asking for it in the main one made the model miss upgrade prices.
LIST_PROMPT = ("This is the player's money in the players list of Roblox Tower Defense Simulator. From $1,000 up it "
               "is shortened, K meaning thousands: $1.21K = 1210, $6.09K = 6090, $27.6K = 27600, $145K = 145000; "
               "multiply by 1000, don't just drop the point. Reply with a JSON array of one integer: the money in "
               "dollars, or -1 if you cannot read it.")
PANELS = (("CASH", "cash"), ("WAVE", "wave"), ("TOWERS", "towers_count"), ("UPGRADE", "upgrade_box"))
HOTBAR_PROMPT = ("This is the tower hotbar of Roblox Tower Defense Simulator: slots 1 to 5 from left to right, each "
                 "showing its price in green under the tower. Ignore the small slot number in each slot's corner, the "
                 "$ sign and commas, and any ability buttons on the far left. Reply with a JSON array of the five "
                 "prices, -1 for an empty slot.")


@dataclass
class Reading:
    cash: int = -1
    wave: int = -1
    total_waves: int = -1
    towers: int = -1
    max_towers: int = -1
    upgrade_cost: int = -1  # next upgrade of the selected tower: price, 0 = fully upgraded, -1 = no panel
    cash2: int = -1  # the cash in the players list (top right), rounded there above $1,000; -1 = not read
    upgrade_maxed: bool = False  # the panel's level bars are all lit: really fully upgraded
    t: float = 0.0  # time.monotonic() when the frame it was read from was captured
    latency_ms: float = 0.0

    def fresh(self, max_age=3.0):
        return bool(self.t) and time.monotonic() - self.t <= max_age


class NumberReader:
    def __init__(self, cfg):
        self.c = cfg["reader"]
        self.url = self.c.get("url", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
        rois = cfg.get("rois", {})
        self.rois = {label: rois[key] for label, key in PANELS if key in rois}
        self.panel_roi = rois.get("upgrade_panel")
        self.hotbar_roi = rois.get("hotbar")
        self.label = f"ollama:{self.c['model']}"
        self.latest = Reading()
        self.error = ""
        self.list_roi = rois.get("cash_list")
        self.fields, self.prompt = FIELDS, PROMPT
        n = len(self.fields)
        self.schema = {"type": "array", "items": {"type": "integer"}, "minItems": n, "maxItems": n}
        self._believed = self._last_raw = None  # cash smoothing, see smooth()
        self._warm = False  # has the model answered yet? Until then, wait long enough for it to load

    def panel_open(self, frame):
        """Is a tower's upgrade panel open? Found by its red Sell button somewhere in the panel area.

        The vision model reads 'no panel' and 'Fully Upgraded' both as 0, so this check decides.
        """
        if self.panel_roi is None:
            return True
        return sell_button_visible(_crop(frame, self.panel_roi))

    def image(self, frame):
        crops = [(label, _crop(frame, roi)) for label, roi in self.rois.items()]
        return np.vstack([_panel(c, label, 320) for label, c in crops if c is not None])

    def _post(self, payload):
        """Ask Ollama. A cold model takes 30-60 s to load, and when the client hangs up first Ollama
        aborts the load and the next request starts it over: with a 10 s timeout it never finished.
        So until the model has answered (and again after an error) the wait is `load_timeout`."""
        if not self._warm:
            print(f"[reader] waiting for {self.c['model']} to load into the GPU (up to a minute the first time)...")
        try:
            answer = _post_json(self.url, payload, {},
                                self.c.get("timeout", 10) if self._warm else self.c.get("load_timeout", 180))
        except BrainError:
            self._warm = False
            raise
        self._warm = True
        return answer

    def read(self, frame):
        t0 = time.perf_counter()
        payload = {
            "model": self.c["model"],
            "stream": False,
            "think": False,
            "keep_alive": self.c.get("keep_alive", "30m"),
            "format": self.schema,
            "options": {"temperature": 0, "num_predict": 48},
            "messages": [{"role": "user", "content": self.prompt, "images": [_jpeg_b64(self.image(frame))]}],
        }
        try:
            msg = self._post(payload)["message"]
        except BrainError as e:
            if "cannot reach" in str(e):
                raise BrainError(f"{e} - is Ollama running? (ollama serve)") from None
            raise
        values = json.loads(msg.get("content") or "[]")
        if not isinstance(values, list) or len(values) != len(self.fields):
            raise ValueError(f"unexpected reader answer: {msg.get('content')!r}")
        reading = Reading(*(_to_int(v) for v in values), latency_ms=(time.perf_counter() - t0) * 1000)
        if not self.panel_open(frame):
            reading.upgrade_cost = -1
        elif "UPGRADE" in self.rois:
            reading.upgrade_maxed = upgrade_bars_full(_crop(frame, self.rois["UPGRADE"]))
        return reading

    def smooth(self, reading):
        """Guard against a misread extra digit (845 read as 8445) without lagging behind spending.

        A drop is believed at once (spending is real, and believing too little cash only makes the bot
        wait). A rise must show in two reads in a row that agree with each other, so misreads
        ($1,307 seen as $15,007, or $5,316 as $55,341 and then $41,341) never reach a decision.
        """
        if reading.cash < 0:
            return reading
        cash2 = self._list_cash(reading.cash2, reading.cash) if reading.cash2 >= 0 else -1
        if cash2 >= 0 and abs(reading.cash - cash2) > max(60, 0.02 * cash2):
            reading.cash = min(reading.cash, cash2)  # the two counters disagree: one is misread
        raw, prev = reading.cash, self._last_raw
        self._last_raw = raw
        if self._believed is None or raw <= self._believed:
            self._believed = raw
        elif prev is not None and abs(raw - prev) <= max(300, 0.15 * raw):
            self._believed = min(raw, prev)
        reading.cash = self._believed
        return reading

    def read_list_cash(self, frame):
        """The cash in the players list (top right), in dollars as the model reads it, or -1."""
        crop = _crop(frame, self.list_roi) if self.list_roi else None
        if crop is None:
            return -1
        payload = {
            "model": self.c["model"], "stream": False, "think": False, "keep_alive": self.c.get("keep_alive", "30m"),
            "format": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 1},
            "options": {"temperature": 0, "num_predict": 12},
            "messages": [{"role": "user", "content": LIST_PROMPT, "images": [_jpeg_b64(_panel(crop, "CASH", 320))]}],
        }
        values = json.loads(self._post(payload)["message"]["content"] or "[]")
        return _to_int(values[0]) if isinstance(values, list) and values else -1

    def needs_check(self, reading):
        """A sudden rise of the cash (more than kill money): worth a second opinion before believing it."""
        base = self._believed if self._believed is not None else 0
        return reading.cash >= 0 and reading.cash - base > max(300, 0.15 * base)

    def reset(self):
        """A new match: forget the cash believed in the last one."""
        self._believed = self._last_raw = None

    def _list_cash(self, cash2, cash):
        """The players-list cash in dollars, or -1 if it fits nothing. The model sometimes reads a
        shortened $6.09K as 609 (or $1.46K as 14600), so a read may be scaled by 10, 100 or 1000 (or a
        tenth): whichever lands within 25% of the bottom counter or of the cash believed so far."""
        refs = [r for r in (cash, self._believed) if r and r > 0]
        best = None
        for k in (range(4) if cash2 < 1000 else (0, -1)):  # $1.46K seen as 146 or as 14600
            c = round(cash2 * 10 ** k)
            for r in refs:
                q = max(c, r) / max(1, min(c, r))
                if q <= 1.25 and (best is None or q < best[0]):
                    best = (q, c)
        return best[1] if best else -1

    def read_hotbar(self, frame):
        """The five tower prices on the hotbar, left to right (-1 for an empty slot)."""
        roi = self.hotbar_roi
        crop = _crop(frame, roi) if roi else None
        if crop is None:
            raise ValueError("no hotbar region in config")
        payload = {
            "model": self.c["model"], "stream": False, "think": False, "keep_alive": self.c.get("keep_alive", "30m"),
            "format": {"type": "array", "items": {"type": "integer"}, "minItems": 5, "maxItems": 5},
            "options": {"temperature": 0, "num_predict": 40},
            "messages": [{"role": "user", "content": HOTBAR_PROMPT, "images": [_jpeg_b64(_panel(crop, "HOTBAR", 640))]}],
        }
        values = json.loads(self._post(payload)["message"]["content"] or "[]")
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError(f"unexpected hotbar answer: {values!r}")
        return [_to_int(v) for v in values]

    def run_forever(self, latest_frame, stop):
        """Thread body: keep self.latest current until `stop` is set."""
        interval = self.c.get("interval", 0.3)
        while not stop.is_set():
            frame, captured = latest_frame()
            if frame is None:
                stop.wait(0.1)
                continue
            try:
                reading = self.read(frame)
                if self.list_roi and self.needs_check(reading):
                    reading.cash2 = self.read_list_cash(frame)
                reading = self.smooth(reading)
                reading.t = captured
                self.latest, self.error = reading, ""
            except (BrainError, ValueError, KeyError, TypeError) as e:
                self.error = str(e)
                print(f"[reader] {e}")
                stop.wait(2.0)
            stop.wait(interval)
