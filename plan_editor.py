"""Build-plan editor: which towers the bot places, in what order, and how far it upgrades each.

Double-click "Edit Plan.bat" (or press P when "Start TDS Bot.bat" asks). Saves into config.json.
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import configio

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
FONT = ("Tahoma", 10)  # has Thai glyphs on every Windows version
AFTER = {"ai": "ให้ AI เล่นต่อเอง / AI keeps playing", "upgrade": "อัปเกรดป้อมที่มีอย่างเดียว / Only upgrade"}


class PlanEditor:
    def __init__(self, root, config_path=CONFIG):
        self.root, self.path = root, Path(config_path)
        self.cfg = configio.load(self.path)
        self.towers = self.cfg.get("towers", {})
        self.names = [t.get("name", f"slot {s}") for s, t in sorted(self.towers.items())]
        root.title(f"TDS Bot - แผนการวางป้อม / Build plan ({self.path.name})")
        root.option_add("*Font", FONT)
        ttk.Style().configure("Treeview", font=FONT, rowheight=24)
        ttk.Style().configure("Treeview.Heading", font=(FONT[0], 10, "bold"))

        frame = ttk.Frame(root, padding=10)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        ttk.Label(frame, text="บอทจะทำทีละขั้นจากบนลงล่าง: วางป้อมทีละตัว และอัปเกรดแต่ละตัวถึงเลเวลที่ตั้งไว้ก่อนวางตัวต่อไป\n"
                              "จำนวน 0 = ไม่วางเพิ่ม แต่อัปเกรดป้อมชนิดนั้นที่มีอยู่ทุกตัวให้ถึงเลเวล\n"
                              "The bot works top to bottom: places each tower, upgrading it to the level before the next.\n"
                              "Count 0 = place none, upgrade every one of that tower already placed.",
                  justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        cols = ("step", "tower", "count", "level", "cost", "waves")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", height=10, selectmode="browse")
        for col, title, width in (("step", "#", 40), ("tower", "ป้อม / Tower", 150), ("count", "จำนวน / Count", 110),
                                  ("level", "อัปเกรดถึงเลเวล / Upgrade to", 190), ("cost", "ราคา / Cost", 100),
                                  ("waves", "วางได้ / Waves", 120)):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="center")
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", lambda _: self.load_selected())
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        side = ttk.Frame(frame)
        side.grid(row=1, column=1, sticky="n", padx=(8, 0))
        for text, cmd in (("▲ ขึ้น / Up", lambda: self.move(-1)), ("▼ ลง / Down", lambda: self.move(1)),
                          ("✕ ลบ / Remove", self.remove), ("ล้างแผน / Clear", self.clear)):
            ttk.Button(side, text=text, command=cmd, width=16).pack(pady=2)

        edit = ttk.LabelFrame(frame, text="ขั้นตอน / Step", padding=8)
        edit.grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)
        self.tower = tk.StringVar(value=self.names[0] if self.names else "")
        self.count = tk.IntVar(value=1)
        self.level = tk.IntVar(value=0)
        ttk.Label(edit, text="ป้อม / Tower").grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Combobox(edit, textvariable=self.tower, values=self.names, state="readonly", width=16).grid(
            row=1, column=0, padx=(0, 16), sticky="w")
        ttk.Label(edit, text="จำนวน / Count (0 = อัปตัวที่มี)").grid(row=0, column=1, sticky="w", padx=(0, 16))
        ttk.Spinbox(edit, textvariable=self.count, from_=0, to=20, width=6).grid(row=1, column=1, padx=(0, 16), sticky="w")
        ttk.Label(edit, text="อัปเกรดถึงเลเวล / Upgrade to (0 = ไม่อัป)").grid(row=0, column=2, sticky="w", padx=(0, 16))
        ttk.Spinbox(edit, textvariable=self.level, from_=0, to=5, width=6).grid(row=1, column=2, padx=(0, 16), sticky="w")
        ttk.Button(edit, text="+ เพิ่มขั้น / Add", command=self.add).grid(row=1, column=3, padx=4)
        ttk.Button(edit, text="✓ แก้ขั้นที่เลือก / Update", command=self.update).grid(row=1, column=4, padx=4)

        bottom = ttk.Frame(frame)
        bottom.grid(row=3, column=0, columnspan=2, sticky="ew")
        ttk.Label(bottom, text="จบแผนแล้ว / After the plan:").pack(side="left")
        self.after = tk.StringVar(value=AFTER.get(self.cfg.get("after_plan", "ai"), AFTER["ai"]))
        ttk.Combobox(bottom, textvariable=self.after, values=list(AFTER.values()), state="readonly",
                     width=34).pack(side="left", padx=6)
        ttk.Button(bottom, text="บันทึก / Save", command=self.save).pack(side="right")
        self.status = ttk.Label(frame, text="", foreground="#2e7d32")
        self.status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.steps = [dict(s) for s in self.cfg.get("plan") or []]
        self.refresh()

    # --- helpers
    def tower_info(self, name):
        for slot, t in self.towers.items():
            if t.get("name", "").lower() == str(name).lower() or slot == str(name):
                return slot, t
        return None, {}

    def refresh(self, select=None):
        self.tree.delete(*self.tree.get_children())
        for i, s in enumerate(self.steps):
            _, t = self.tower_info(s["tower"])
            cost = f"${t['cost']:,}" if t.get("cost") else "-"
            first, last = t.get("from_wave"), t.get("until_wave")
            waves = (f"{first or 1}-{last}" if last else f"{first}+" if first else "ตลอด / any")
            level = s.get("upgrade_to", 0) or "-"
            count = f"x{s.get('count', 1)}" if s.get("count", 1) else "อัปที่มี / existing"
            self.tree.insert("", "end", iid=str(i), values=(i + 1, s["tower"], count, level, cost, waves))
        if select is not None and 0 <= select < len(self.steps):
            self.tree.selection_set(str(select))
            self.tree.see(str(select))

    def selected(self):
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def load_selected(self):
        i = self.selected()
        if i is not None:
            s = self.steps[i]
            self.tower.set(s["tower"])
            self.count.set(s.get("count", 1))
            self.level.set(s.get("upgrade_to", 0))

    def current_step(self):
        try:
            count, level = int(self.count.get()), int(self.level.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Build plan", "จำนวนและเลเวลต้องเป็นตัวเลข / Count and level must be numbers")
            return None
        return {"tower": self.tower.get(), "count": max(0, count), "upgrade_to": max(0, level)}

    # --- actions
    def add(self):
        step = self.current_step()
        if step:
            i = self.selected()
            at = len(self.steps) if i is None else i + 1
            self.steps.insert(at, step)
            self.refresh(select=at)
            self.status.config(text="")

    def update(self):
        i, step = self.selected(), self.current_step()
        if i is not None and step:
            self.steps[i] = step
            self.refresh(select=i)

    def move(self, delta):
        i = self.selected()
        if i is not None and 0 <= i + delta < len(self.steps):
            self.steps[i], self.steps[i + delta] = self.steps[i + delta], self.steps[i]
            self.refresh(select=i + delta)

    def remove(self):
        i = self.selected()
        if i is not None:
            del self.steps[i]
            self.refresh(select=min(i, len(self.steps) - 1))

    def clear(self):
        if self.steps and messagebox.askyesno("Build plan", "ลบทุกขั้น? ไม่มีแผน = AI เลือกเองทั้งเกม\n"
                                                            "Remove every step? No plan = the AI chooses all game."):
            self.steps = []
            self.refresh()

    def save(self):
        cfg = configio.load(self.path)  # re-read: calibration may have changed spots meanwhile
        actions = cfg.setdefault("actions", [])
        for s in self.steps:  # a planned tower must be placeable
            slot, _ = self.tower_info(s["tower"])
            action = f"PLACE_TOWER_{slot}"
            if slot and action not in actions:
                at = next((k for k, a in enumerate(actions) if not a.startswith("PLACE_TOWER_")), len(actions))
                actions.insert(at, action)
        cfg["plan"] = self.steps
        cfg["after_plan"] = next(k for k, v in AFTER.items() if v == self.after.get())
        configio.save(self.path, cfg)
        self.status.config(text=f"บันทึกแล้ว {len(self.steps)} ขั้น ใช้ตอนเริ่มบอทครั้งถัดไป / Saved {len(self.steps)} steps "
                                f"- used the next time the bot starts")


def main():
    root = tk.Tk()
    PlanEditor(root, sys.argv[1] if len(sys.argv) > 1 else CONFIG)  # a config path, e.g. config_minigunner.json
    root.mainloop()


if __name__ == "__main__":
    main()
