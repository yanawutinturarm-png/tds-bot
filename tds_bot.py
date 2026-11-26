"""Tower Defense Simulator bot: screen capture -> AI decision -> Roblox input.

Threads:
  main      captures the Roblox window with mss and draws the OpenCV preview
  brain     sends the latest frame to the AI backend, vets the answer, performs it
  failsafe  polls the keyboard globally: Q pauses/resumes, ESC kills

Run `python tds_bot.py --help` for options.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

if sys.platform != "win32":
    sys.exit("This bot only runs on Windows (Roblox + DirectInput).")

import cv2
import mss
import numpy as np

from brain import BrainError, Decision, make_brain
from reader import NumberReader, Reading
from recorder import HumanRecorder
from endgame import Endgame
from controls import (ActionExecutor, Control, Failsafe, GameWindow, apply_loadout, enable_dpi_awareness,
                      hold_single_instance, keep_awake, match_loadout, parse_plan, validate_config)

ROOT = Path(__file__).resolve().parent
PREVIEW_WINDOW = "TDS Bot - preview"
AUTOSTART_DELAY = 3.0

GREEN, YELLOW, ORANGE, RED = (80, 220, 80), (0, 220, 255), (0, 150, 255), (60, 60, 255)
WHITE, GREY = (255, 255, 255), (170, 170, 170)


class Shared:
    """State handed between the capture/preview loop and the brain thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self.region = None
        self.decision = None
        self.result = ""
        self.ai_input = None
        self.decisions = 0
        self.phase = "starting"
        self.brain_label = ""
        self.autostart_at = None  # monotonic time at which --autostart begins playing
        self.reader = None        # NumberReader, when exact numbers are read
        self._captured = 0.0

    def publish(self, frame, region):
        with self._lock:
            self._frame, self.region, self._captured = frame, region, time.monotonic()

    def latest_frame(self):
        with self._lock:
            return self._frame

    def latest_frame_with_time(self):
        with self._lock:
            return self._frame, self._captured

    def reading(self):
        """The latest exact numbers, or an empty Reading when there is no fresh one."""
        r = self.reader.latest if self.reader else None
        return r if r and r.fresh() else Reading()


# ------------------------------------------------------------------- brain

def check_loadout(reader, frame, executor, cfg, tries):
    """Read the hotbar prices and re-key the towers to match the real hotbar order.

    Returns None when done (matched, or given up after 8 tries), else the number of tries so far.
    """
    stamp = time.strftime("%H:%M:%S")
    try:
        prices = reader.read_hotbar(frame)
    except (BrainError, ValueError) as e:
        prices = []
        print(f"[{stamp}] hotbar: {e}")
    mapping = match_loadout(prices, cfg["towers"]) if prices else None
    if mapping is None:
        if tries + 1 >= 8:
            print(f"[{stamp}] hotbar prices {prices} don't match the towers in config.json - "
                  f"using the configured order. Check the loadout in config.json.")
            return None
        return tries + 1
    if any(slot != src for slot, src in mapping.items()):
        apply_loadout(cfg, mapping)
        executor.plan = parse_plan(cfg)
    order = ", ".join(f"{slot} {t['name']}" for slot, t in sorted(cfg["towers"].items()))
    print(f"[{stamp}] hotbar: {order}")
    return None


def note_of(notes, observed):
    return "; ".join(n for n in notes + [observed] if n)


def brain_worker(brain, executor, shared, control, cfg):
    min_interval = cfg["brain"].get("min_interval", 0.35)
    loadout_tries = 0 if shared.reader and cfg.get("check_loadout", True) else None
    last_endgame = None
    log = open(ROOT / cfg["log_file"], "a", encoding="utf-8") if cfg.get("log_file") else None
    shared.phase = "loading model"
    try:
        brain.warmup()
        shared.phase = "ready"
    except Exception as e:
        shared.phase = "warm-up failed"
        print(f"[brain] warm-up failed: {e}")
    try:
        while not control.stop.is_set():
            frame = shared.latest_frame()
            if control.paused or frame is None:
                control.stop.wait(0.05)
                continue
            t0 = time.monotonic()
            notes = [executor.press_ready(frame, shared.region)]  # TDS waits for READY before wave 1
            restarted = executor.press_restart(frame, shared.region)  # end of a match: play again
            if restarted:
                notes.append(restarted)
                if shared.reader and hasattr(shared.reader, "reset"):
                    shared.reader.reset()
            for n in notes:
                if n:
                    print(f"[{time.strftime('%H:%M:%S')}] {n}")
            executor.track_camera(frame)  # a boss skill can shake or shift the view
            if restarted:
                if log:
                    log.write(json.dumps({"t": time.time(), "action": "RESTART", "result": restarted, "reason": "",
                                          "cash": -1, "wave": -1, "towers": -1, "latency_ms": 0, "source": "",
                                          "note": restarted, "bot_towers": []}) + "\n")
                    log.flush()
                continue  # this tick's frame is the old match's result screen: decide nothing from it
            reading = shared.reading()
            if shared.reader and not reading.fresh():
                shared.phase = "waiting for the number reader"  # never act on cash it hasn't seen
                control.stop.wait(0.2)
                continue
            if loadout_tries is not None:  # once per run: which tower is on which hotbar key
                loadout_tries = check_loadout(shared.reader, frame, executor, cfg, loadout_tries)
                if loadout_tries is not None:
                    control.stop.wait(0.5)
                    continue
            observed = executor.observe(reading)  # spending, upgrade panel, last placement
            used = executor.use_abilities(frame, shared.region)  # Commander: Call to Arms F, Support Caravan B
            observed = "; ".join(n for n in (observed, used) if n) or None
            if observed:
                print(f"[{time.strftime('%H:%M:%S')}] {observed}")
            if executor.human_turn:  # the player plays this wave; the bot only watches and records
                shared.phase = "the player's turn - recording"
                control.stop.wait(0.2)
                continue
            if executor.endgame is not None:  # waves 34-35: sell, top levels, Strongest, leapfrog Wardens
                eg = executor.endgame.tick(frame, reading, shared.region)
                if eg is not None:
                    shared.phase = "endgame"
                    shared.result = eg
                    acted = eg.endswith(" done") or "done" in eg.split(": ")[-1]
                    if acted or eg != last_endgame:
                        print(f"[{time.strftime('%H:%M:%S')}] {eg}")
                        if log:
                            log.write(json.dumps({"t": time.time(), "action": "ENDGAME", "result": eg, "reason": "",
                                                  "cash": executor.cash, "wave": reading.wave, "towers": reading.towers,
                                                  "latency_ms": 0, "source": "endgame", "note": note_of(notes, observed),
                                                  "boss": executor.endgame.progress,
                                                  "bot_towers": [executor.tower_name(t["slot"]) for t in executor.placed]}) + "\n")
                            log.flush()
                    last_endgame = eg
                    control.stop.wait(max(0.0, min_interval - (time.monotonic() - t0)))
                    continue
            note = "; ".join(n for n in notes + [observed] if n)
            ctx = executor.context()
            ctx.update(cash=executor.cash, wave=reading.wave, total_waves=reading.total_waves)
            try:
                decision = brain.decide(frame, ctx)
                shared.phase = "ready"
            except Exception as e:
                decision = Decision(reason=f"brain error: {e}", source=brain.label)
                shared.phase = "error, retrying"
                control.stop.wait(1.0)
            if decision.cash < 0 and reading.cash >= 0:  # brains that can't read numbers get the reader's
                decision.cash, decision.wave = reading.cash, reading.wave
            if control.should_abort():
                continue  # paused or killed while the model was thinking: drop the stale decision
            result = executor.execute(decision)
            shared.decision, shared.result = decision, result
            shared.ai_input = brain.last_input
            shared.decisions += 1
            print(f"[{time.strftime('%H:%M:%S')}] {decision.action:<16} -> {result:<38} "
                  f"{decision.latency_ms:5.0f} ms  cash={decision.cash} wave={decision.wave}  {decision.reason}")
            if log:
                log.write(json.dumps({"t": time.time(), "action": decision.action, "result": result,
                                      "reason": decision.reason, "cash": decision.cash, "wave": decision.wave,
                                      "towers": reading.towers, "latency_ms": round(decision.latency_ms),
                                      "source": decision.source, "note": note,
                                      "bot_towers": [executor.tower_name(t["slot"]) for t in executor.placed]}) + "\n")
                log.flush()
            control.stop.wait(max(0.0, min_interval - (time.monotonic() - t0)))
    finally:
        if log:
            log.close()


# ----------------------------------------------------------------- preview

def _text(img, text, org, color=WHITE, scale=0.5, boxed=False):
    text = str(text).encode("ascii", "replace").decode("ascii")  # OpenCV fonts are ASCII-only
    if boxed:  # dark backing so labels stay readable over the game
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (org[0] - 2, org[1] - h - 3), (org[0] + w + 2, org[1] + base), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_preview(frame, cfg, shared, control, executor, window, fps, dry_run):
    pw = cfg["preview"].get("width", 960)
    vh = max(1, round(frame.shape[0] * pw / frame.shape[1]))
    view = cv2.resize(frame, (pw, vh), interpolation=cv2.INTER_AREA)

    for name, (x, y, w, h) in cfg.get("rois", {}).items():
        p0, p1 = (int(x * pw), int(y * vh)), (int((x + w) * pw), int((y + h) * vh))
        cv2.rectangle(view, p0, p1, YELLOW, 1)
        _text(view, name, (p0[0] + 3, p0[1] + 14), YELLOW, 0.4, boxed=True)

    used = {t["spot"]: t["slot"] for t in executor.placed}
    for i, (fx, fy) in enumerate(executor.spots):
        c = (int(fx * pw), int(fy * vh))
        cv2.circle(view, c, 7, GREEN if i in used else WHITE, -1 if i in used else 1)
        label = f"{i + 1} {executor.tower_name(used[i])}" if i in used else str(i + 1)
        _text(view, label, (c[0] + 9, c[1] + 4), WHITE, 0.4, boxed=True)

    panel_h = 120
    shade = view.copy()
    cv2.rectangle(shade, (0, 0), (pw, panel_h), (0, 0, 0), -1)
    cv2.rectangle(shade, (0, vh - 24), (pw, vh), (0, 0, 0), -1)
    view = cv2.addWeighted(shade, 0.6, view, 0.4, 0)

    hk = cfg["hotkeys"]
    if control.stop.is_set():
        state, color = "STOPPED", RED
    elif executor.human_turn:
        state, color = f"YOUR TURN (wave {executor.handover_wave}) - the bot records how you play; {hk['kill']} stops it", ORANGE
    elif control.paused and shared.autostart_at:
        left = max(0.0, shared.autostart_at - time.monotonic())
        state, color = f"STARTING IN {left:.0f} s - {hk['pause']} pauses, {hk['kill']} kills", YELLOW
    elif control.paused:
        state, color = f"PAUSED - focus Roblox and press {hk['pause']} to run", YELLOW
    elif not dry_run and not window.is_focused():
        state, color = "RUNNING - inputs held until Roblox is focused", ORANGE
    else:
        state, color = "RUNNING", GREEN
    flags = (["DRY RUN"] if dry_run else []) + ([] if window.found else ["Roblox window not found: fixed region"])
    _text(view, state + (f"   [{' | '.join(flags)}]" if flags else ""), (10, 22), color, 0.6)

    d = shared.decision or Decision()
    _text(view, f"Brain {shared.brain_label} ({shared.phase})   latency {d.latency_ms:.0f} ms   "
                f"decisions {shared.decisions}   capture {fps:.0f} fps", (10, 46))
    _text(view, f"Last AI action: {d.action}   {d.reason[:60]}", (10, 70), WHITE, 0.55)
    _text(view, f"Result: {shared.result[:90]}", (10, 92), GREY)
    r = shared.reading()
    cash = f"{max(d.cash, r.cash):,}" if max(d.cash, r.cash) >= 0 else "?"
    wave = f"{r.wave}/{r.total_waves}" if r.wave >= 0 else (d.wave if d.wave >= 0 else "?")
    towers = f"{r.towers}/{r.max_towers} in game, " if r.towers >= 0 else ""
    _text(view, f"Cash ${cash}   Wave {wave}   Towers {towers}{len(executor.placed)} placed by the bot", (10, 113))
    _text(view, f"{hk['pause']}: pause/resume    {hk['kill']}: kill    mouse to top-left corner: kill    "
                f"(hotkeys are global)", (10, vh - 8), GREY, 0.45)

    ai = shared.ai_input
    if cfg["preview"].get("show_ai_input", True) and ai is not None:
        tw = pw // 4
        th = min(round(ai.shape[0] * tw / ai.shape[1]), vh - panel_h - 40)
        if th > 20:
            x0, y0 = pw - tw - 6, panel_h + 6
            view[y0:y0 + th, x0:x0 + tw] = cv2.resize(ai, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.rectangle(view, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), GREY, 1)
            _text(view, "what the AI sees", (x0, y0 + th + 14), GREY, 0.4, boxed=True)
    return view


# -------------------------------------------------------------------- main

def use_reader(cfg):
    """The ollama brain reads numbers itself; every other brain gets the number reader when it is enabled."""
    return cfg.get("reader", {}).get("enabled", False) and cfg["brain"].get("backend") != "ollama"


def run(cfg, args):
    dry_run = args.dry_run or cfg["safety"].get("dry_run", False)
    control = Control(start_paused=args.autostart or cfg["safety"].get("start_paused", True))
    shared = Shared()
    window = GameWindow(cfg["capture"])
    window.refresh()
    executor = ActionExecutor(cfg, control, window, lambda: shared.region, dry_run)
    brain = make_brain(cfg)
    shared.brain_label = brain.label

    def on_pause_key():
        if executor.human_turn:
            return  # the player's turn: Q is the game's cancel key now, not the bot's pause
        shared.autostart_at = None  # Q during the countdown takes over from auto-start
        control.toggle_pause()

    hk = cfg["hotkeys"]
    failsafe = Failsafe(hk["pause"], hk["kill"], on_pause_key, control.kill)
    failsafe.start()
    if not dry_run and cfg["safety"].get("keep_awake", True):
        keep_awake(True)  # unattended runs: no sleep, no screen-off while playing
    if cfg.get("endgame") and cfg.get("path"):  # waves 34-35 the way the player won
        executor.endgame = Endgame(cfg, executor, shared.latest_frame)
        eg = cfg["endgame"]
        final = {"lose": "sells everything so the match ends lost (a win would end in the lobby)",
                 "idle": "is not played", "win": "is played to win: Crook Bosses on Strongest, Wardens leapfrog the Warlord"}
        print(f"Endgame from wave {eg.get('from_wave', 34)}: sell Farms, top levels; wave {eg.get('boss_wave', 35)} "
              f"{final.get(eg.get('final_wave', 'lose'), '')}.")
    recorder = None
    if cfg.get("handover", {}).get("wave"):  # the player plays from that wave; the bot records how
        recorder = HumanRecorder(cfg, shared.latest_frame, lambda: shared.region, window.is_focused,
                                 executor._from_screen, shared.reading)
        executor.recorder = recorder
        recorder.start()
        print(f"Wave {cfg['handover']['wave']}: the bot hands over to you and records how you play "
              f"(recordings/). Nobody playing within {executor.handover_wait_s:.0f} s: it plays on.")
    worker = threading.Thread(target=brain_worker, name="brain", daemon=True,
                              args=(brain, executor, shared, control, cfg))
    worker.start()
    if use_reader(cfg):
        shared.reader = NumberReader(cfg)
        executor.reader_on = not dry_run  # read upgrade prices before pressing upgrade (no panel opens in a dry run)
        threading.Thread(target=shared.reader.run_forever, name="reader", daemon=True,
                         args=(shared.latest_frame_with_time, control.stop)).start()
        print(f"Number reader: {shared.reader.label} (cash, wave, tower counter, upgrade panel)")

    preview = cfg["preview"].get("enabled", True) and not args.no_preview
    frame_dt = 1.0 / cfg["capture"].get("fps", 30)
    print(f"Brain: {brain.label}   dry-run: {dry_run}   Roblox window found: {window.found}")
    print(f"Capturing {window.region}")
    if args.autostart:
        shared.autostart_at = time.monotonic() + AUTOSTART_DELAY
        print(f"Auto-start: playing in {AUTOSTART_DELAY:.0f} s.  {hk['pause']} pauses, {hk['kill']} kills the bot.")
    else:
        print(f"{'Starting PAUSED: focus Roblox, then press ' + hk['pause'] + ' to run.' if control.paused else 'RUNNING.'}"
              f"  {hk['kill']} kills the bot.")

    fps, next_refresh, focus_pending = 0.0, 0.0, args.autostart and not dry_run
    refocus_s = 0 if dry_run else cfg["safety"].get("refocus_after_s", 60)
    unfocused_since = None
    try:
        with getattr(mss, "MSS", mss.mss)() as sct:  # mss.mss is deprecated from mss 10.2
            while not control.stop.is_set():
                t0 = time.perf_counter()
                if t0 >= next_refresh:  # the window can move or resize
                    window.refresh()
                    next_refresh = t0 + 1.0
                    # Unattended: a popup that took the focus would stop the bot for good. Running (not
                    # paused) and Roblox unfocused for a while: bring it back.
                    if refocus_s and not control.paused and window.found and not window.is_focused():
                        unfocused_since = unfocused_since or t0
                        if t0 - unfocused_since >= refocus_s:
                            unfocused_since = t0
                            print(f"[{time.strftime('%H:%M:%S')}] Roblox lost the focus {refocus_s:.0f} s ago; bringing it back")
                            window.focus()
                    else:
                        unfocused_since = None
                region = window.region
                frame = cv2.cvtColor(np.asarray(sct.grab(region)), cv2.COLOR_BGRA2BGR)
                shared.publish(frame, region)
                if preview:
                    cv2.imshow(PREVIEW_WINDOW, render_preview(frame, cfg, shared, control, executor,
                                                              window, fps, dry_run))
                    cv2.waitKey(1)
                    if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        control.kill()  # closing the preview window stops the bot
                if focus_pending:  # after the preview exists, so it doesn't take the focus back
                    focus_pending = False
                    if not window.focus():
                        print("Click the Roblox window: the bot only plays while Roblox is focused.")
                if shared.autostart_at and time.monotonic() >= shared.autostart_at:
                    shared.autostart_at = None
                    control.resume()
                elapsed = time.perf_counter() - t0
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
                fps = 0.9 * fps + 0.1 / (time.perf_counter() - t0)
    except KeyboardInterrupt:
        pass
    finally:
        control.kill()
        failsafe.halt()
        if recorder:
            recorder.halt()
        keep_awake(False)
        cv2.destroyAllWindows()
        worker.join(timeout=3)
        print(f"Bot stopped. Matches finished this run: {executor.matches}" + (" (see matches.jsonl)" if executor.matches else ""))


def test_brain(cfg, image_path):
    """Run the brain on a saved screenshot: tune the prompt/ROIs without touching the game."""
    frame = cv2.imread(image_path)
    if frame is None:
        sys.exit(f"Cannot read image {image_path!r}")
    brain = make_brain(cfg)
    print(f"Brain {brain.label}: loading model...")
    t0 = time.perf_counter()
    try:
        brain.warmup()
    except BrainError as e:
        sys.exit(f"Brain error: {e}")
    print(f"  ready in {time.perf_counter() - t0:.1f}s")
    if not cfg.get("placement_spots"):  # not calibrated yet: pretend there is room, so placing can be tested
        cfg = {**cfg, "placement_spots": [[0.05 * i, 0.5] for i in range(1, 11)]}
    executor = ActionExecutor(cfg, Control(False), GameWindow(cfg["capture"]), lambda: None, dry_run=True)
    r = Reading()
    if use_reader(cfg):
        reader = NumberReader(cfg)
        try:
            r = reader.read(frame)
        except BrainError as e:
            sys.exit(f"Number reader error: {e}")
        print(f"  reader {reader.label}: cash={r.cash} wave={r.wave}/{r.total_waves} "
              f"towers={r.towers}/{r.max_towers}  ({r.latency_ms:.0f} ms)")
        r.t = time.monotonic()
        executor.observe(r)
    ctx = executor.context()
    ctx.update(cash=r.cash, wave=r.wave, total_waves=r.total_waves)
    blocked = [f"{a}: {why}" for a in cfg["actions"] if (why := executor._blocked(a))]
    if blocked:
        print("  not offered: " + "; ".join(blocked))
    for i in range(3):
        d = brain.decide(frame, ctx)
        afford = f" affordable={d.affordable}" if d.affordable else ""
        print(f"  #{i + 1}: {d.latency_ms:5.0f} ms -> {d.action:<16} cash={d.cash} wave={d.wave}{afford}  {d.reason}")
    if brain.last_input is not None:
        out = ROOT / "ai_input_debug.jpg"
        cv2.imwrite(str(out), brain.last_input)
        print(f"  image sent to the AI saved as {out}")


def probe_test(cfg, slot):
    """Show where the bot thinks a tower can go, without placing anything.

    Brings Roblox to the front, presses the tower key, sweeps the preview over every calibrated
    spot and its nearest ring, then puts the tower back. Saves probe_test.png: green = ok,
    red = blocked, grey = hidden. Needs a match in progress. ESC aborts.
    """
    from controls import Aborted, RobloxInput
    from placement import Prober

    window = GameWindow(cfg["capture"])
    window.refresh()
    if not window.found:
        sys.exit("Roblox window not found.")
    spots = cfg.get("placement_spots", [])
    if not spots:
        sys.exit("No placement_spots yet: run calibrate.py --only spots first.")
    control = Control(start_paused=False)
    failsafe = Failsafe(cfg["hotkeys"]["pause"], cfg["hotkeys"]["kill"], control.kill, control.kill)
    failsafe.start()
    inp = RobloxInput(cfg.get("input", {}), lambda: control.should_abort() or not window.is_focused())
    prober = Prober({**cfg, "placement": {**cfg.get("placement", {}), "search_px": cfg.get("placement", {}).get(
        "search_px", [0, 28])[:2]}})
    region, key = window.region, cfg["keys"][f"tower_{slot}"]
    print(f"Probing {len(spots)} spots with tower slot {slot} ({key}). Don't touch the mouse; ESC aborts.")
    window.focus()
    time.sleep(1.0)
    results, saw = [], False
    try:
        park = (region["left"] + region["width"] // 2, region["top"] + region["height"] // 5)
        reference = prober.begin(inp, key, park, region)
        for i, (fx, fy) in enumerate(spots):
            if i:
                reference = prober.rebase(inp, park, region)
            for x, y in prober.candidates(fx, fy, region, panel_open=False):
                verdict, stats = prober.test(inp, x, y, region, reference)
                saw |= verdict != "hidden"
                results.append((i, x, y, verdict, stats))
        if saw:
            inp.press(key)  # put the tower back
    except Aborted:
        print("Aborted.")
    finally:
        failsafe.halt()
    frame = prober.grab(region)
    colors = {"ok": (80, 220, 80), "blocked": (60, 60, 255), "hidden": (160, 160, 160)}
    for i, x, y, verdict, _ in results:
        cv2.circle(frame, (x - region["left"], y - region["top"]), 5, colors[verdict], -1)
    for i, (fx, fy) in enumerate(spots):
        c = (round(fx * region["width"]), round(fy * region["height"]))
        _text(frame, str(i + 1), (c[0] + 8, c[1] - 8), WHITE, 0.5, boxed=True)
    out = ROOT / "probe_test.png"
    cv2.imwrite(str(out), frame)
    counts = Counter(v for *_, v, _ in results)
    per_spot = {i: [v for j, *_, v, _ in results if j == i] for i in range(len(spots))}
    print(f"{counts['ok']} ok, {counts['blocked']} blocked, {counts['hidden']} hidden; "
          f"spots with no ok position: {[i + 1 for i, v in per_spot.items() if v and 'ok' not in v]}")
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description="AI bot for Roblox Tower Defense Simulator")
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--backend", choices=["jev", "ollama", "openai", "rules"], help="override brain.backend")
    p.add_argument("--model", help="override the model of the chosen backend")
    p.add_argument("--dry-run", action="store_true", help="decide and log, but never send any input")
    p.add_argument("--no-preview", action="store_true", help="no OpenCV window")
    p.add_argument("--test-brain", metavar="IMAGE", help="run the brain on a screenshot and exit")
    p.add_argument("--probe-test", metavar="SLOT", nargs="?", const="2",
                   help="sweep a tower preview over the spots without placing, save probe_test.png (default slot 2)")
    p.add_argument("--autostart", action="store_true",
                   help=f"bring Roblox to the front and start playing after {AUTOSTART_DELAY:.0f} s, no pause key needed")
    args = p.parse_args()
    # Model text can contain emoji; never let a console code page (e.g. cp874) crash the brain thread.
    sys.stdout.reconfigure(errors="replace")

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    if args.backend:
        cfg["brain"]["backend"] = args.backend
    if args.model:
        cfg["brain"].setdefault(cfg["brain"]["backend"], {})["model"] = args.model
    validate_config(cfg)
    enable_dpi_awareness()  # before any capture or input, so all coordinates are physical pixels

    if not args.test_brain and not hold_single_instance():
        raise SystemExit("Another TDS Bot is already running (its preview window is open). Close it first (ESC): "
                         "two bots on one game fight over the mouse and keys.")
    if args.test_brain:
        test_brain(cfg, args.test_brain)
    elif args.probe_test:
        probe_test(cfg, args.probe_test)
    else:
        run(cfg, args)


if __name__ == "__main__":
    main()
