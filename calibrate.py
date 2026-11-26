"""Mark the UI regions, tower placement spots and (optionally) the skip-wave button
on a screenshot of your game. Everything is saved to config.json as fractions of the
game window, so it keeps working if you resize Roblox.

  python calibrate.py                 # capture Roblox after a 3 s countdown
  python calibrate.py --image shot.png
  python calibrate.py --only spots    # rois | spots | skip
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np

import configio
from controls import client_region, enable_dpi_awareness, find_window, focus_window

ROOT = Path(__file__).resolve().parent
ROI_NAMES = {
    "cash": "the cash / money counter",
    "cash_list": "your cash in the players list at the top right (e.g. $600), a second opinion on the cash",
    "towers_count": "the tower counter under the cash (placed / max, e.g. 7/20)",
    "wave": "the wave counter",
    "hotbar": "the tower hotbar",
    "upgrade_panel": "the upgrade panel (only visible with a tower selected)",
    "upgrade_box": "the green next-upgrade box inside the panel (price or 'Fully Upgraded!')",
    "sell_button": "a small piece of the red Sell button, left of its text (tells the bot the panel is open)",
    "ready_button": "the green READY button shown before wave 1 (only on that screen; press C elsewhere)",
}
MAX_DISPLAY_WIDTH = 1280


def capture(cfg, delay, focus=False):
    title = cfg["capture"].get("window_title", "Roblox")
    hwnd = find_window(title) if cfg["capture"].get("auto_find_window", True) else None
    region = client_region(hwnd)
    if region is None:
        region = cfg["capture"]["region"]
        print(f"Roblox window not found, using the fixed region {region}")
    if focus and focus_window(hwnd):
        print(f"Roblox brought to the front - capturing in {delay}s...")
    else:
        print(f"Bring Roblox to the front - capturing in {delay}s...")
    time.sleep(delay)
    with getattr(mss, "MSS", mss.mss)() as sct:  # mss.mss is deprecated from mss 10.2
        return cv2.cvtColor(np.asarray(sct.grab(region)), cv2.COLOR_BGRA2BGR)


def pick_rois(disp, rois):
    dh, dw = disp.shape[:2]
    for name, what in ROI_NAMES.items():
        guide = disp.copy()
        if name in rois:
            x, y, w, h = rois[name]
            cv2.rectangle(guide, (int(x * dw), int(y * dh)), (int((x + w) * dw), int((y + h) * dh)), (0, 220, 255), 1)
        title = f"Drag a box around {name.upper()} - ENTER confirm, C keep current"
        print(f"{name}: drag a box around {what}. ENTER/SPACE confirms, C keeps the current box (yellow).")
        x, y, w, h = cv2.selectROI(title, guide, showCrosshair=False)
        cv2.destroyWindow(title)
        if w and h:
            rois[name] = [round(x / dw, 4), round(y / dh, 4), round(w / dw, 4), round(h / dh, 4)]
    return rois


def pick_points(disp, points, title, help_text, single=False):
    dh, dw = disp.shape[:2]
    original = list(points)
    points = list(points)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            if single:
                points.clear()
            points.append([round(x / dw, 4), round(y / dh, 4)])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    print(help_text)
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        view = disp.copy()
        for i, (fx, fy) in enumerate(points):
            c = (int(fx * dw), int(fy * dh))
            cv2.circle(view, c, 8, (80, 220, 80), 2)
            cv2.putText(view, str(i + 1), (c[0] + 10, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):        # ENTER / SPACE: save
            break
        if key == 27:              # ESC: discard changes
            points = original
            break
        if key in (ord("r"), ord("R")):
            points.clear()
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(title)
    return points


def main():
    p = argparse.ArgumentParser(description="Calibrate TDS bot regions and placement spots")
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--image", help="calibrate on a saved screenshot instead of capturing")
    p.add_argument("--only", choices=["rois", "spots", "skip"], help="run just one step")
    p.add_argument("--delay", type=float, default=3, help="seconds before capturing")
    p.add_argument("--focus", action="store_true", help="bring Roblox to the front before capturing")
    args = p.parse_args()

    enable_dpi_awareness()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image {args.image!r}")
    else:
        frame = capture(cfg, args.delay, args.focus)
        cv2.imwrite(str(ROOT / "calibration.png"), frame)
        print(f"Screenshot saved as calibration.png ({frame.shape[1]}x{frame.shape[0]})")

    scale = min(1.0, MAX_DISPLAY_WIDTH / frame.shape[1])
    disp = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    steps = [args.only] if args.only else ["rois", "spots", "skip"]
    if "rois" in steps:
        cfg["rois"] = pick_rois(disp, dict(cfg.get("rois", {})))
    if "spots" in steps:
        cfg["placement_spots"] = pick_points(
            disp, cfg.get("placement_spots", []),
            "Placement spots - L-click add, R-click undo, R clear, ENTER save, ESC cancel",
            "Placement spots: left-click every spot where the bot may place a tower (in build order). "
            "Right-click undoes, R clears, ENTER saves, ESC cancels.")
    if "skip" in steps:
        current = [cfg["skip_wave_button"]] if cfg.get("skip_wave_button") else []
        picked = pick_points(
            disp, current, "Skip-wave button - click it, ENTER save, R clear (use key instead)",
            "Skip wave: click the on-screen skip button if TDS needs a click instead of the skip key. "
            "R clears (use keys.skip_wave), ENTER saves, ESC cancels.", single=True)
        cfg["skip_wave_button"] = picked[0] if picked else None

    with open(args.config, "w", encoding="utf-8") as f:
        f.write(configio.dumps(cfg) + "\n")
    print(f"Saved {args.config}: {len(cfg.get('rois', {}))} regions, {len(cfg.get('placement_spots', []))} spots, "
          f"skip button {'set' if cfg.get('skip_wave_button') else 'not set'}.")


if __name__ == "__main__":
    main()
