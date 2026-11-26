"""Decision backends: turn a game frame into exactly one bot action.

Backends:
  jev     - Jev-style decision model over the System One API (/v1/systemone): a probability
            for every action instead of generated text. Free by default through a local
            OneJev server; can also target TypeSafe's hosted Jev (paid).
  ollama  - local vision LLM through Ollama's /api/chat (free, offline)
  openai  - any OpenAI-compatible /chat/completions endpoint with image input
  rules   - no AI at all; a fixed place/upgrade loop for testing the pipeline
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

SYSTEM_PROMPT = """You control a bot playing the Roblox game Tower Defense Simulator (TDS).
The image is made of labeled panels cropped from the game screen (for example CASH, WAVE,
HOTBAR, UPGRADE_PANEL) plus a small FULL VIEW of the map.
First read the cash and wave numbers, then choose exactly ONE allowed action.
Answer with a single JSON object and nothing else:
{"cash": <integer, -1 if unreadable>, "wave": <integer, -1 if unreadable>, "action": "<ACTION>", "reason": "<at most 10 words>"}"""

ACTION_HELP = {
    "UPGRADE_SELECTED": "upgrade one of your placed towers (the bot picks which, round-robin)",
    "SKIP_WAVE": "vote to skip ahead to the next wave",
    "SELL": "sell the selected tower for some cash back",
    "WAIT": "do nothing this tick (save cash)",
}


@dataclass
class Decision:
    action: str = "WAIT"
    reason: str = ""
    cash: int = -1
    wave: int = -1
    latency_ms: float = 0.0
    source: str = ""
    raw: str = ""
    affordable: dict = field(default_factory=dict)  # tower slot -> bool, for brains that can't read numbers


class BrainError(RuntimeError):
    pass


# ---------------------------------------------------------------- image prep

def _crop(frame, roi):
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = max(0, int(x * w)), max(0, int(y * h))
    x1, y1 = min(w, int((x + rw) * w)), min(h, int((y + rh) * h))
    crop = frame[y0:y1, x0:x1]
    return crop if crop.size else None


def _panel(img, label, width):
    scale = width / img.shape[1]
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    body = cv2.resize(img, (width, max(1, round(img.shape[0] * scale))), interpolation=interp)
    bar = np.full((22, width, 3), 32, np.uint8)
    cv2.putText(bar, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, body])


# Small helper regions for the number reader; the AI sees them inside the bigger panels anyway.
NOT_FOR_AI = {"sell_button", "upgrade_box", "ready_button"}


def build_ai_image(frame, rois, col_width=448, include_full=True):
    """Stack the labeled UI crops (and a small full view) into one image."""
    panels = []
    for name, roi in rois.items():
        if name in NOT_FOR_AI:
            continue
        crop = _crop(frame, roi)
        if crop is not None:
            panels.append(_panel(crop, name.upper(), col_width))
    if include_full or not panels:
        panels.append(_panel(frame, "FULL VIEW", col_width))
    if len(panels) == 1:
        return panels[0]
    # Two columns keep the image close to square, which vision models resize best.
    cols, heights = [[], []], [0, 0]
    for p in sorted(panels, key=len, reverse=True):
        i = heights.index(min(heights))
        cols[i].append(p)
        heights[i] += len(p)
    h = max(heights)
    return np.hstack([np.vstack(c + [np.zeros((h - hh, col_width, 3), np.uint8)])
                      for c, hh in zip(cols, heights)])


def _jpeg_b64(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise BrainError("could not encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------- prompt / parsing

def place_description(slot, towers):
    t = towers.get(slot, {})
    cost = f" (costs ${t['cost']})" if t.get("cost") else ""
    note = f": {t['note']}" if t.get("note") else ""
    return f"place {t.get('name', f'the slot {slot} tower')} from hotbar slot {slot}{cost}{note}"


def build_prompt(actions, towers, hint, ctx):
    lines = ["Allowed actions:"]
    for a in actions:
        if a.startswith("PLACE_TOWER_"):
            lines.append(f"- {a}: {place_description(a[-1], towers)}")
        else:
            lines.append(f"- {a}: {ACTION_HELP.get(a, '')}")
    lines.append(f"Bot state: {ctx['placed']} of {ctx['spots']} placement spots used "
                 f"(towers placed: {ctx.get('towers', 'none')}), {ctx['upgrades']} upgrades bought so far.")
    if ctx.get("history"):
        lines.append("Recent actions: " + ", ".join(ctx["history"]))
    if hint:
        lines.append("Strategy: " + hint)
    lines.append("Read CASH and WAVE from the image, then reply with the JSON object only.")
    return "\n".join(lines)


def _to_int(value):
    digits = re.sub(r"[^\d]", "", str(value)) if value is not None else ""
    if str(value).strip().startswith("-") or not digits:
        return -1
    return int(digits)


def parse_decision(raw, actions):
    data = {}
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    action = str(data.get("action", "")).strip().upper()
    if action not in actions:
        # Small models sometimes break the format; take the first action name they mention.
        upper = raw.upper()
        action = next((a for a in sorted(actions, key=len, reverse=True) if a in upper), "WAIT")
    return Decision(action=action, reason=str(data.get("reason", ""))[:80],
                    cash=_to_int(data.get("cash")), wave=_to_int(data.get("wave")), raw=raw)


def _post_json(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise BrainError(f"HTTP {e.code} from {url}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise BrainError(f"cannot reach {url}: {getattr(e, 'reason', e)}") from None


# ------------------------------------------------------------------ backends

class RulesBrain:
    """No AI: alternate placing towers and upgrading them. Ignores the frame."""

    def __init__(self, cfg):
        self.actions = cfg["actions"]
        self.label = "rules"
        self.last_input = None
        self._tick = 0

    def warmup(self):
        pass

    def decide(self, frame, ctx):
        self._tick += 1
        place = [a for a in self.actions if a.startswith("PLACE_TOWER_")]
        if place and ctx["placed"] < ctx["spots"] and (ctx["placed"] == 0 or self._tick % 2):
            return Decision(action=place[ctx["placed"] % len(place)], reason="fill next spot", source=self.label)
        if ctx["placed"] and "UPGRADE_SELECTED" in self.actions:
            return Decision(action="UPGRADE_SELECTED", reason="spend on upgrades", source=self.label)
        return Decision(reason="nothing to do", source=self.label)


class _VisionBrain:
    def __init__(self, cfg, section):
        b = cfg["brain"]
        self.c = b[section]
        self.actions = cfg["actions"]
        self.towers = cfg.get("towers", {})
        self.rois = cfg.get("rois", {})
        self.hint = cfg.get("strategy_hint", "")
        self.col_width = b.get("image_width", 448)
        self.include_full = b.get("include_full_frame", True)
        self.timeout = self.c.get("timeout", 15)
        self.max_tokens = self.c.get("max_tokens", 96)
        self.label = f"{section}:{self.c['model']}"
        self.last_input = None
        # Field order matters: reading the numbers first gives the model a tiny "look before you act" step.
        self.schema = {
            "type": "object",
            "properties": {
                "cash": {"type": "integer"},
                "wave": {"type": "integer"},
                "action": {"type": "string", "enum": list(self.actions)},
                "reason": {"type": "string"},
            },
            "required": ["cash", "wave", "action", "reason"],
        }

    def _call(self, image_b64, prompt, timeout):
        raise NotImplementedError

    def warmup(self):
        """Load the model before the first real decision (the first call can take many seconds)."""
        tiny = _jpeg_b64(np.zeros((64, 64, 3), np.uint8))
        self._call(tiny, 'Reply with {"cash": -1, "wave": -1, "action": "WAIT", "reason": "warmup"}',
                   max(self.timeout, 180))

    def decide(self, frame, ctx):
        t0 = time.perf_counter()
        img = build_ai_image(frame, self.rois, self.col_width, self.include_full)
        self.last_input = img
        actions = [a for a in self.actions if a in ctx.get("feasible", self.actions)] or ["WAIT"]
        raw = self._call(_jpeg_b64(img), build_prompt(actions, self.towers, self.hint, ctx), self.timeout)
        d = parse_decision(raw, self.actions)
        d.latency_ms = (time.perf_counter() - t0) * 1000
        d.source = self.label
        return d


class OllamaBrain(_VisionBrain):
    def __init__(self, cfg):
        super().__init__(cfg, "ollama")
        self.url = self.c.get("url", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"

    def _call(self, image_b64, prompt, timeout):
        payload = {
            "model": self.c["model"],
            "stream": False,
            "keep_alive": self.c.get("keep_alive", "30m"),
            "format": self.schema,
            "options": {"temperature": 0.2, "num_predict": self.max_tokens},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": [image_b64]},
            ],
        }
        if self.c.get("think") is not None:  # false keeps thinking models (Gemma 4, Qwen3) fast
            payload["think"] = self.c["think"]
        msg = _post_json(self.url, payload, {}, timeout)["message"]
        if not msg.get("content") and msg.get("thinking"):
            raise BrainError(f"{self.c['model']} used its whole token budget thinking; "
                             'set "think": false under brain.ollama in config.json')
        return msg.get("content", "")


class OpenAICompatBrain(_VisionBrain):
    """Any OpenAI-style /chat/completions API with image input (Jev AI, LM Studio, vLLM, ...)."""

    def __init__(self, cfg):
        super().__init__(cfg, "openai")
        self.url = self.c["base_url"].rstrip("/") + "/chat/completions"
        self.api_key = os.environ.get(self.c.get("api_key_env", ""), "") or self.c.get("api_key", "")

    def _call(self, image_b64, prompt, timeout):
        payload = {
            "model": self.c["model"],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
        }
        if self.c.get("json_mode", True):
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = _post_json(self.url, payload, headers, timeout)
        return resp["choices"][0]["message"]["content"] or ""


class JevBrain:
    """Typed decisions over the System One API (POST /v1/systemone).

    Instead of generating text, a Jev-style model returns a calibrated probability for every
    allowed action, plus yes/no probabilities for the guard questions (in a match? can I
    afford tower N?). Works with a local OneJev server (free, `qev serve`) or TypeSafe's
    hosted Jev (https://api.typesafe.ai, paid, images still experimental there).
    """

    def __init__(self, cfg):
        b = cfg["brain"]
        self.c = b["jev"]
        self.actions = cfg["actions"]
        self.towers = cfg.get("towers", {})
        self.rois = cfg.get("rois", {})
        self.hint = cfg.get("strategy_hint", "")
        self.col_width = b.get("image_width", 448)
        self.include_full = b.get("include_full_frame", True)
        self.timeout = self.c.get("timeout", 10)
        self.min_confidence = self.c.get("min_confidence", 0.25)
        self.hosted = "typesafe.ai" in self.c["url"]
        self.url = self.c["url"].rstrip("/") + "/v1/systemone"
        self.api_key = os.environ.get(self.c.get("api_key_env", ""), "")
        self.label = "jev:typesafe" if self.hosted else "jev:onejev"
        self.last_input = None

    def _costs(self, actions):
        """Distinct prices of the towers that could be placed right now, cheapest first."""
        return sorted({self.towers.get(a[-1], {}).get("cost", 0) for a in actions if a.startswith("PLACE_TOWER_")} - {0})

    def _questions(self, actions, costs):
        criteria = {a: place_description(a[-1], self.towers) if a.startswith("PLACE_TOWER_") else ACTION_HELP.get(a, a)
                    for a in actions}
        questions = {
            "in_match": {"type": "noul", "instructions": "The screen shows a Tower Defense Simulator match in progress, "
                                                         "not a lobby, menu, loading or results screen."},
            "action": {"type": "choice", "instructions": "Which single action should the bot take right now?",
                       "criteria": criteria},
        }
        for cost in costs:
            questions[f"cash_{cost}"] = {"type": "noul", "instructions": f"The CASH panel shows at least ${cost:,}."}
        return questions

    @staticmethod
    def _cash_floor(answers, costs):
        """Turn the per-price yes/no answers into one consistent cash level.

        Cash is a single number, so "at least $X" must be yes for every price up to some level
        and no above it. Pick the level the answers support best (maximum likelihood); this
        fixes contradictions such as yes for $2,250 but no for $1,850.
        """
        p = [min(max(answers[f"cash_{c}"]["noul"], 1e-4), 1 - 1e-4) for c in costs]
        scores = [sum(math.log(x) for x in p[:k]) + sum(math.log(1 - x) for x in p[k:]) for k in range(len(p) + 1)]
        k = max(range(len(scores)), key=scores.__getitem__)
        return costs[k - 1] if k else 0

    def _call(self, img, state, questions, timeout):
        b64 = _jpeg_b64(img)
        payload = {"model": self.c.get("model", "jev-latest"), "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if self.hosted:  # TypeSafe: images travel as assets, behind an opt-in header
            state["screen"] = "tds-screen"
            payload["assets"] = {"tds-screen": {"type": "image", "format": "jpg", "data_base64": b64}}
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        else:            # OneJev: images travel as media, referenced by <image:N>
            state["screen"] = "<image:1>"
            payload["media"] = [{"type": "image", "data": f"data:image/jpeg;base64,{b64}"}]
        try:
            return _post_json(self.url, payload, headers, timeout)["answers"]
        except BrainError as e:
            if not self.hosted and "cannot reach" in str(e):
                raise BrainError(f"{e} - is the OneJev server running? Start start_onejev.bat") from None
            raise

    def warmup(self):
        blank = np.zeros((320, 320, 3), np.uint8)  # hosted Jev needs images of at least 280 px per side
        self._call(blank, {"note": "warm-up"}, {"ready": {"type": "noul", "instructions": "The image is black."}},
                   max(self.timeout, 300))

    def decide(self, frame, ctx):
        t0 = time.perf_counter()
        img = build_ai_image(frame, self.rois, self.col_width, self.include_full)
        self.last_input = img
        state = {
            "game": "Roblox Tower Defense Simulator. The image stacks labeled crops of the game UI "
                    "(CASH, WAVE, HOTBAR, UPGRADE_PANEL) and a small FULL VIEW of the map.",
            "bot": f"{ctx['placed']} of {ctx['spots']} placement spots used "
                   f"(towers placed: {ctx.get('towers', 'none')}), {ctx['upgrades']} upgrades bought",
            "recent_actions": ", ".join(ctx.get("history", [])) or "none",
            "strategy": self.hint,
        }
        # Only offer what is possible right now (no upgrade before a tower exists, no tower past its limit).
        actions = [a for a in self.actions if a in ctx.get("feasible", self.actions)] or ["WAIT"]
        cash = ctx.get("cash", -1)
        if cash >= 0:
            # Exact cash from the number reader: only offer towers we can pay for, and say so.
            offered = [a for a in actions if a.startswith("PLACE_TOWER_")]
            actions = [a for a in actions if not a.startswith("PLACE_TOWER_")
                       or cash >= self.towers.get(a[-1], {}).get("cost", 0)]
            buyable = [a for a in actions if a.startswith("PLACE_TOWER_")]
            if buyable:
                # Spend before skipping: calling the next wave early only pays when there is nothing to buy.
                actions = [a for a in actions if a != "SKIP_WAVE"]
                # Waiting only makes sense to save up; if every offered tower is affordable, there is nothing to save for.
                if len(buyable) == len(offered):
                    actions = [a for a in actions if a != "WAIT"]
            state["cash"] = f"${cash:,}"
            if ctx.get("wave", -1) >= 0:
                state["wave"] = f"{ctx['wave']} of {ctx['total_waves']}" if ctx.get("total_waves", -1) > 0 else ctx["wave"]
            costs = []
        else:
            costs = self._costs(actions)  # no reader: estimate cash with yes/no questions per tower price
        if ctx.get("must") and len(actions) > 1:  # the build plan's next move is possible: do it
            actions = [a for a in actions if a != "WAIT"]
        if actions == ["WAIT"]:  # nothing to choose (e.g. saving up): no need to ask
            return Decision(reason=ctx.get("saving") or "nothing affordable to do", source=self.label,
                            cash=cash, wave=ctx.get("wave", -1), latency_ms=(time.perf_counter() - t0) * 1000)
        answers = self._call(img, state, self._questions(actions, costs), self.timeout)
        floor = self._cash_floor(answers, costs) if costs else None
        affordable = {a[-1]: self.towers.get(a[-1], {}).get("cost", 0) <= (cash if cash >= 0 else floor)
                      for a in self.actions if a.startswith("PLACE_TOWER_") and (cash >= 0 or floor is not None)}
        # Drop placements the cash can't cover, then share their probability among what is left.
        probs = {a: p for a, p in answers["action"]["probabilities"].items()
                 if a in actions and affordable.get(a[-1] if a.startswith("PLACE_TOWER_") else "", True)}
        total = sum(probs.values()) or 1.0
        ranked = sorted(((a, p / total) for a, p in probs.items()), key=lambda kv: kv[1], reverse=True) or [("WAIT", 1.0)]
        top, p = ranked[0]
        runner = f", next {ranked[1][0]} {ranked[1][1]:.2f}" if len(ranked) > 1 else ""
        cash_note = f", cash>=${floor:,}" if floor else (", cash<cheapest tower" if floor == 0 else "")
        in_match = answers.get("in_match", {}).get("noul", 1.0)
        counting = ctx.get("wave", -1) >= 1 and ctx.get("total_waves", -1) >= 1  # the reader sees a wave running
        if in_match < 0.5 and not counting:
            action, reason = "WAIT", f"not in a match (p={in_match:.2f})"
        elif p < self.min_confidence:
            action, reason = "WAIT", f"unsure: {top} p={p:.2f}{runner}{cash_note}"
        else:
            action, reason = top, f"p={p:.2f}{runner}{cash_note}"
        return Decision(action=action, reason=reason, affordable=affordable, source=self.label,
                        cash=cash, wave=ctx.get("wave", -1),
                        latency_ms=(time.perf_counter() - t0) * 1000, raw=json.dumps(answers)[:800])


def make_brain(cfg):
    backend = cfg["brain"].get("backend", "ollama")
    if backend == "jev":
        return JevBrain(cfg)
    if backend == "ollama":
        return OllamaBrain(cfg)
    if backend == "openai":
        return OpenAICompatBrain(cfg)
    if backend == "rules":
        return RulesBrain(cfg)
    raise SystemExit(f"Unknown brain backend {backend!r} (use jev, ollama, openai or rules)")
