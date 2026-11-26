"""Read and write config.json, keeping short lists such as [0.4, 0.76, 0.2, 0.07] on one line."""
from __future__ import annotations

import json
import re
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dumps(cfg):
    text = json.dumps(cfg, indent=2, ensure_ascii=False)
    text = re.sub(r"\[\s+([-\d.,\s]+?)\s+\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text)
    # Plan steps on one line each: {"tower": "Soldier", "count": 4, "upgrade_to": 2}
    return re.sub(r"\{\s+(\"tower\"[^{}]*?)\s+\}", lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)) + "}", text)


def save(path, cfg):
    Path(path).write_text(dumps(cfg) + "\n", encoding="utf-8")
