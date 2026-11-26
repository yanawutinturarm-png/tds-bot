"""One-click start for the TDS bot (what "Start TDS Bot.bat" runs).

1. makes sure Ollama runs with the number-reader model (gemma3:4b), pulling it once if needed
2. starts the free local Jev brain (OneJev) in the background, unless it already runs
3. waits until Roblox is open
4. the first time, or when C is pressed, lets you mark the tower spots for the map
5. starts the bot, which brings Roblox to the front and plays after a short countdown
6. stops the Jev brain and unloads the reader model when the bot exits, freeing the GPU

Arguments go on to the bot, e.g.  "Start TDS Bot.bat" --dry-run
"""
from __future__ import annotations

import json
import msvcrt
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from controls import bot_running, client_region, find_window

ROOT = Path(__file__).resolve().parent
ONEJEV_PY = ROOT / ".venv-onejev" / "Scripts" / "python.exe"
SERVER_LOG = ROOT / "onejev_server.log"
C_KEYS = "cแ"  # the C key on English and Thai keyboard layouts
P_KEYS = "pย"  # the P key on English and Thai keyboard layouts


def say(msg):
    print(msg, flush=True)


def server_up(url, path="/health"):
    try:
        with urllib.request.urlopen(url + path, timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def start_jev_brain(cfg):
    """Start OneJev unless the bot doesn't use it or it already runs. Returns the process started here."""
    url = cfg["brain"]["jev"]["url"].rstrip("/")
    if cfg["brain"].get("backend") != "jev" or "typesafe.ai" in url:
        return None
    if server_up(url):
        say("สมอง Jev ทำงานอยู่แล้ว")
        return None
    if not ONEJEV_PY.exists():
        raise SystemExit("ยังไม่ได้ติดตั้งสมอง Jev: รัน setup.bat ก่อน")
    say("กำลังเปิดสมอง Jev ในเครื่อง (ประมาณ 30 วินาที)...")
    # Shares this console: closing the window or Ctrl+C stops it too. Its output goes to onejev_server.log.
    log = open(SERVER_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen([str(ONEJEV_PY), str(ROOT / "onejev_server.py"), "--port", url.rsplit(":", 1)[-1]],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    t0 = time.monotonic()
    while not server_up(url):
        if proc.poll() is not None or time.monotonic() - t0 > 600:
            proc.kill()
            log.close()
            tail = SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            raise SystemExit("เปิดสมอง Jev ไม่สำเร็จ (ดู onejev_server.log):\n" + "\n".join(tail))
        time.sleep(1)
    say(f"สมอง Jev พร้อมแล้ว ({time.monotonic() - t0:.0f} วินาที)")
    return proc


def ollama_model(cfg):
    """The Ollama model this run needs (number reader, or the ollama brain), or None."""
    if cfg["brain"].get("backend") == "ollama":
        return cfg["brain"]["ollama"]["url"].rstrip("/"), cfg["brain"]["ollama"]["model"]
    if cfg.get("reader", {}).get("enabled"):
        return cfg["reader"]["url"].rstrip("/"), cfg["reader"]["model"]
    return None


def ensure_ollama(cfg):
    """Make sure Ollama runs and has the model; returns (url, model) to unload at the end, or None."""
    need = ollama_model(cfg)
    if not need:
        return None
    url, model = need
    if not server_up(url, "/api/tags"):
        say("กำลังเปิด Ollama...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        except FileNotFoundError:
            raise SystemExit("ไม่พบ Ollama: ติดตั้งจาก https://ollama.com แล้วลองใหม่") from None
        for _ in range(60):
            if server_up(url, "/api/tags"):
                break
            time.sleep(1)
        else:
            raise SystemExit("เปิด Ollama ไม่สำเร็จ")
    with urllib.request.urlopen(url + "/api/tags", timeout=5) as r:
        names = {m["name"] for m in json.load(r)["models"]}
    if model not in names and f"{model}:latest" not in names:
        say(f"กำลังดาวน์โหลด {model} สำหรับอ่านตัวเลข (ครั้งเดียว ประมาณ 3 GB)...")
        subprocess.run(["ollama", "pull", model], check=True)
    return need


def warm_ollama(need, keep_alive):
    """Load the reader model into the GPU in the background while the Jev brain starts. A cold load
    can take a minute (from a slow disk); the bot's first reads would otherwise wait for it."""
    url, model = need
    body = json.dumps({"model": model, "keep_alive": keep_alive}).encode()

    def load():
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/api/generate", data=body, method="POST",
                                                          headers={"Content-Type": "application/json"}),
                                   timeout=300).close()
        except OSError as e:
            say(f"โหลด {model} ไม่สำเร็จ ({e}) บอทจะลองเองอีกครั้ง")

    worker = threading.Thread(target=load, name="warm-reader", daemon=True)
    worker.start()
    return worker


def unload_ollama(need):
    """Free the reader model's GPU memory now instead of after keep_alive runs out."""
    url, model = need
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url + "/api/generate", data=body, method="POST",
                                                      headers={"Content-Type": "application/json"}), timeout=10).close()
    except OSError:
        pass


def wait_for_roblox(title):
    if client_region(find_window(title)):
        return
    say("ยังไม่เจอ Roblox: เปิด Tower Defense Simulator ไว้ แล้วบอทจะไปต่อเอง (Ctrl+C = ยกเลิก)")
    while not client_region(find_window(title)):
        time.sleep(2)
    say("เจอ Roblox แล้ว")


def pressed_within(keys, seconds):
    """The first of `keys` typed within `seconds`, or None."""
    while msvcrt.kbhit():  # ignore keys typed earlier
        msvcrt.getwch()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if msvcrt.kbhit() and (key := msvcrt.getwch().lower()) in keys:
            return key
        time.sleep(0.05)
    return None


def edit_plan(config_path):
    say("เปิดหน้าต่างแผนการวางป้อม: แก้แล้วกด Save แล้วปิดหน้าต่างเพื่อเริ่มบอท")
    subprocess.run([sys.executable, str(ROOT / "plan_editor.py"), str(config_path)], cwd=ROOT)


def mark_spots(config_path):
    say("มาร์กจุดวางป้อม: เข้าแมตช์ใน Roblox ให้เห็นแมพ แล้วกด Enter ที่หน้าต่างนี้")
    input()
    subprocess.run([sys.executable, str(ROOT / "calibrate.py"), "--config", str(config_path),
                    "--only", "spots", "--focus", "--delay", "1"], cwd=ROOT)


def main():
    sys.stdout.reconfigure(errors="replace")
    args = sys.argv[1:]
    config_path = Path(args[args.index("--config") + 1]) if "--config" in args else ROOT / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if bot_running():
        say("บอทอีกตัวเปิดอยู่แล้ว (หน้าต่าง TDS Bot - preview) ปิดตัวเก่าก่อน: กด ESC หรือปิดหน้าต่างนั้น"
            " / Another bot is already running: close it first (ESC).")
        if sys.stdin.isatty():
            input("กด Enter เพื่อปิด / Press Enter to close ")
        return
    brain = reader = None
    try:
        reader = ensure_ollama(cfg)
        warming = warm_ollama(reader, cfg.get("reader", {}).get("keep_alive", "30m")) if reader else None
        brain = start_jev_brain(cfg)
        wait_for_roblox(cfg["capture"].get("window_title", "Roblox"))
        if sys.stdin.isatty():
            if not cfg.get("placement_spots"):
                say("ยังไม่มีจุดวางป้อม (ครั้งแรก หรือแมพใหม่)")
                mark_spots(config_path)
            say("ภายใน 3 วินาที: C = มาร์กจุดวางป้อมใหม่ (เปลี่ยนแมพ), P = แก้แผนการวางป้อม, ไม่กด = เริ่มเลย")
            key = pressed_within(C_KEYS + P_KEYS, 3)
            if key and key in C_KEYS:
                mark_spots(config_path)
            elif key and key in P_KEYS:
                edit_plan(config_path)
        if warming and warming.is_alive():
            say(f"รอโหลดตัวอ่านตัวเลข ({reader[1]}) เข้าการ์ดจอ...")
            warming.join(300)
        say("เริ่มบอท: เล่นเองใน 3 วินาที | Q = หยุด/เล่นต่อ | ESC = ปิดบอท")
        subprocess.run([sys.executable, str(ROOT / "tds_bot.py"), "--autostart", *args], cwd=ROOT)
    except KeyboardInterrupt:
        pass
    finally:
        if reader:
            unload_ollama(reader)
        if brain and brain.poll() is None:
            say("กำลังปิดสมอง Jev...")
            brain.terminate()
            try:
                brain.wait(10)
            except subprocess.TimeoutExpired:
                brain.kill()


if __name__ == "__main__":
    main()
