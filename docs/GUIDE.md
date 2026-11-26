# TDS Bot: full guide

The complete manual: every setting, how each part works, and what was learned along the way.
For the short version see the [README](../README.md).

A local Python bot that watches Roblox with fast screen capture and asks an AI what to do.
It then presses the keys and clicks the mouse through DirectInput, so Roblox registers the input.
**Everything runs free on your own PC.**

```
 Roblox window ──mss──▶ frame ─┬─▶ number reader (gemma3:4b, own thread): cash $18,386, wave 27/40, towers 12/20
                               │                                   │
                               └─▶ crop UI ─▶ AI brain: Jev (local OneJev) ◀──┘ exact numbers + what is possible
                                                        │  P(PLACE_TOWER_1)=0.45, P(PLACE_TOWER_3)=0.38, in match? 0.95 ...
                                                        ▼
   safety vetting (confidence, cooldowns, free spots, cash ≥ cost, tower windows, Roblox focused)
                                                        │
                                                        ▼
      placing: hold the tower, read the game's red no-go shading, click the nearest clear point to the spot
                     pydirectinput: 1-5 / E / X + mouse clicks ──▶ tower counter confirms each placement
```

Actions the AI chooses from: `PLACE_TOWER_1`..`4`, `UPGRADE_SELECTED`, `WAIT`. Also available but off in
`config.json`:

| action | why it is off |
|---|---|
| `PLACE_TOWER_5` | Farm isn't needed at the difficulty played |
| `SKIP_WAVE` | TDS's own auto-skip is on |
| `SELL` | the small AI sold towers whenever cash ran low, then bought them back in a loop |

Before wave 1 the bot also clicks TDS's green **READY** button by itself (`auto_ready`).

## Quick start: one click

Open Roblox, then double-click **`Start TDS Bot.bat`**. It:

1. **Installs everything the first time** via `setup.bat`. PyTorch is several GB, so this takes a while, once.
2. **Prepares the number reader.** It makes sure [Ollama](https://ollama.com) runs and has `gemma3:4b`,
   pulling it once (~3 GB) if needed. Ollama itself must be installed.
3. **Starts the free Jev brain** in the background (~30 s).
4. **Waits** until Roblox is open.
5. **Marks tower spots.** The first time, or when you press **C** within 3 s after a map change, it asks you to
   enter a match and click the spots where towers may go.
6. **Starts the bot.** It brings Roblox to the front and **plays by itself after a 3 s countdown**, clicking
   READY on the pre-game screen.
7. **Cleans up.** When the bot stops, it shuts the Jev brain down and unloads `gemma3:4b` to free the GPU.

While it plays: **Q** pauses and resumes, **ESC** stops the bot. Closing the console window stops everything.

**Choosing what it builds:** double-click **`Edit Plan.bat`**, or press **P** in the 3 s window at start. See
[Build plan](#build-plan).
To watch decisions without any input, run `"Start TDS Bot.bat" --dry-run` from a terminal.
**Minigunner loadout:** `Start TDS Bot - Minigunner.bat` plays the same way with `config_minigunner.json`
(no Farm, Minigunners fill the last tower slots); see
[Second loadout](#second-loadout-minigunners-late-start-tds-bot---minigunnerbat).

The sections below cover the same steps by hand and every setting.

## The AI brains

| backend | what it is | cost | speed* |
|---|---|---|---|
| **`jev`** (default) | Jev-style **decision model**. Instead of writing text, it returns a probability for every action. Runs on a local [OneJev](https://github.com/OmniJev/OneJev) server (Apache-2.0) through the same System One API as [TypeSafe's Jev](https://docs.typesafe.ai). A decision model can't output numbers, so the **number reader** (`gemma3:4b`) reads cash, wave and the tower counter for it. | free | **~0.12 s** (reader ~0.4-0.7 s, in parallel) |
| `ollama` | Local vision LLM (`gemma3:4b`) that reads cash and wave itself and answers in JSON. | free | ~0.45 s |
| `openai` | Any OpenAI-compatible vision API. | depends | network |
| `rules` | No AI: a fixed place-then-upgrade loop, for testing clicks and keys. | free | 0 |

\* Per decision on an RTX 5070, measured on a test frame.

> **About Jev:** Jev is TypeSafe AI's "System One" model. It answers typed questions
> (pick one option, yes/no, score) with calibrated probabilities instead of generating text.
> The hosted Jev costs money, and images there are still marked experimental. So the bot uses
> **OneJev**, a free open model that speaks the same API and was trained on screenshots.
> There is no `jev-ai` package on PyPI. TypeSafe's own SDK is `typesafe-sdk`, and `jev` is a decorator built on it.

---

## 1. Setup

Requirements: Windows 10/11, Python 3.10+, an NVIDIA GPU (for the `jev` and `ollama` brains), and Roblox.

**The bot:**

```bat
cd tds-bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Free local Jev (OneJev server).** This goes in its own venv because PyTorch is several GB:

```bat
python -m venv .venv-onejev
.venv-onejev\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv-onejev\Scripts\python -m pip install "qev[torch] @ git+https://github.com/OmniJev/OneJev.git@80d4e5f2f6dd5af55ca78e74d0dfa3536276106d"
.venv-onejev\Scripts\python -m pip install "triton-windows<3.8" flash-linear-attention
```

- `cu128` builds are needed for RTX 50-series (Blackwell) cards.
- `qev` is pinned to the commit that was reviewed.
- `triton-windows` and `flash-linear-attention` are optional speed-ups.
- The first start downloads `OmniJev/OneJev1.6-0.8B` (about 2.2 GB) into your Hugging Face cache.

**The number reader** (and the `ollama` brain): install [Ollama](https://ollama.com), then `ollama pull gemma3:4b`.
The one-click launcher pulls it for you. With the Jev brain, Roblox and the reader together, a 12 GB GPU was at
about 10 GB.

## 2. Configure the Roblox window

- **Use the primary monitor.** pydirectinput's absolute mouse coordinates only cover the primary display.
- **Any resolution works.** The bot finds the window titled `Roblox` and captures its client area, without borders or title bar.
  All regions and spots are stored as *fractions* of that area. Calibrate at the size you will play at.
- **Windows display scaling** (125 %, 150 %) is fine, because the bot is DPI-aware.
- **Fixed region instead:** set `capture.auto_find_window` to `false` and fill in `capture.region` (physical pixels).
- **Keep the preview window off the game.** Use a second monitor or a corner outside the capture area.
- **Check your TDS keybinds** against `keys` in `config.json`. Defaults: `1`-`5` towers, `e` upgrade, `x` sell
  (both shown as badges in TDS's upgrade panel), `f` skip wave (not confirmed). If skipping is an on-screen button
  rather than a key, mark it during calibration.
- Run Roblox and the bot at the **same privilege level**. If Roblox runs as admin, Windows blocks input from a normal terminal.

## 3. Calibrate (once per map or UI layout)

The default `rois` already match TDS's UI at 1920×1080: wave top-left of center, cash next to the hotbar
bottom-right, upgrade panel bottom-center. They were measured from a real match. **Placement spots** are
different on every map, so those always need calibrating.

Start a match, select one of your towers so the upgrade panel is visible, then run:

```bat
python calibrate.py
```

You get 3 seconds to bring Roblox to the front. A screenshot is saved as `calibration.png`, then:

1. **Regions.** Drag a box around `cash`, `towers_count` (the `7/20` counter under the cash), `wave`, `hotbar` and
   `upgrade_panel`. ENTER confirms, C keeps the current box. Leave room to the right of the cash for 5-6 digit amounts.
2. **Placement spots.** Left-click every spot where towers may go, in build order. Right-click undoes, R clears, ENTER saves.
   The spots don't have to be exact: when placing, the bot looks for a valid position up to ~84 px around each
   spot (see [Placing towers](#placing-towers)). Spread them out along the path rather than packing them tightly.
3. **Skip button (optional).** Click it if TDS needs a click to skip. Press R then ENTER to keep using the `f` key.

Re-run one step with `--only rois|spots|skip`, or calibrate from a file with `--image shot.png`.

Then check the loadout in `config.json`. It is set up for solo Molten (see [Molten notes](#molten-notes)):

| slot | tower | cost | limit | waves | role |
|---|---|---|---|---|---|
| 1 | Soldier | $400 | 4 | until 14 | first waves, then the hidden Demons (wave 11+): level 2 sees hidden enemies, level 3 is 20 DPS; spots 10, 12, 13 |
| 2 | Farm | $300 | 5 | until 20 | cash; spots 18 and 17 on the left sand, the rest right next to the first Farm (`cluster`) |
| 3 | Crook Boss | $2,250 | 5 | any | the most damage on a passing boss (range 22 plus goons); sees hidden enemies from level 2; spots 1, 2, 3, 5, 6, 9, 4 |
| 4 | Commander | $650 | 6 | any | towers in its range fire 10-20% faster; spots 19 (top-right group) and 20 (bottom loop) |
| 5 | Warden | $1,850 | 12 | any | melee, placed right against the path (`hug_path`); spots 11, 15, 8, 16, 14, then free spots nearest the other Wardens |

All five prices differ, so the hotbar check tells every tower apart. (Earlier loadouts had Minigunner in slot 3,
the same $1,850 as the Warden, and Ranger in slot 5, which the bot never got onto a cliff.)

**Spot numbers run from best to worst.** The calibrated spots are numbered by how much path lies within a
Minigunner's range (spot 1 the most); no point on the map does better than spot 1 (21,437 path pixels against
21,430). Spots 14, 15 and 16 were moved to the best melee points by the path bends, measured from the game's own
red shading: 15 and 14 inside the bottom loop (where three Farms used to stand), 16 inside the first bend after the
cave. With 11, 8 and 7 in the top-right loop, the six Wardens spread over three bends, so one Warlord slam can't
reach them all. Spots 19 and 20 are the Commander points, picked to reach the most guns (Wardens counted triple):
19 reaches Wardens 7, 8, 11 and Minigunners 1-4 (at level 3 also 5, 6, 9 and Warden 16), 20 reaches Wardens 14, 15
and Minigunners 6 and 9. (Points on the truck, the barn and the rocks scored well too but are scenery.)
Each tower type takes spots from its own `spots` list first, in that order; when those are used up, it takes
spots no other tower type lists, including new ones grown around placed towers. After calibrating new spots,
ask for them to be sorted the same way.

**The hotbar order doesn't have to match.** At the start of each run the bot reads the five prices on the hotbar
and works out which tower is on which key (`check_loadout`). The prices are unique, so they identify the towers.
It prints e.g. `hotbar: 1 Soldier, 2 Farm, 3 Crook Boss, 4 Commander, 5 Warden`. If the prices don't match
the towers in `config.json` (a different loadout), it says so and uses the configured order.

What gets built in which order is set by the [build plan](#build-plan). A tower is only bought when its
`PLACE_TOWER_N` is in `actions`; the plan editor adds the ones the plan needs.

| field | meaning |
|---|---|
| `name`, `note` | shown to the AI so it knows what each tower is for |
| `cost` | placement price; the bot never places a tower it can't afford |
| `limit` | most copies the bot places (0 or missing = no limit). The game's own limit, or lower to make it buy other towers. |
| `cliff` | `true` for towers that can only stand on cliffs (Ranger). The bot then finds a spot itself, see [Placing towers](#placing-towers). |
| `from_wave`, `until_wave` | the waves in which the bot may place this tower, e.g. Farms only early when they still pay back. Needs the number reader. |
| `save_for` | *only without a build plan:* once this tower's waves start, the bot buys nothing else until it has this many of it |
| `upgrade_to` | *only without a build plan:* after placing one of these, the bot upgrades that one to this level before buying anything else |
| `upgrade_costs` | the tower's upgrade prices in order (Farm `[200, 600, 1250, 2500, 4500]`, Minigunner `[350, 1500, 6500, 21500]`, Crook Boss `[1500, 2750, 14000, 47500]`). Each price belongs to one level, so the panel's price tells the bot the tower's real level, and a price of another tower type means the click opened the wrong tower. A price not in any list (a discount) is used as before. |
| `max_level` | the tower's top level (Farm 5, Minigunner 4, Crook Boss 4). Below it, a panel that reads "Fully Upgraded" is checked twice more before the bot believes it, because a message box over the panel also reads that way. |
| `spots` | optional list of spot numbers (as shown in the preview) this tower uses first, in that order, e.g. `"spots": [5, 6, 7]`. Spots in one tower's list are reserved for it; when its own are used up (or without a list) a tower takes spots no other tower type lists, and new spots grown around placed towers. |
| `cluster` | `true` for towers that don't need the path (Farm): once its own `spots` are used, the next one goes to the clear point nearest the first one of its kind, within `placement.cluster_px` (250 px), so it doesn't take a spot a gun could use. The game draws no red ring around Farms, so the bot keeps 40 px clear of every tower it placed itself. |
| `hug_path` | `true` for melee towers (Warden): the bot aims at the piece of path nearest the spot and places the tower at the clear point right against it. |

Change the loadout, `actions` (`PLACE_TOWER_1`..`5`) and `strategy_hint` together when you switch towers.

## Build plan

The easiest way to steer the bot. Double-click **`Edit Plan.bat`**. It opens a window listing the steps. Each step has:
- **tower:** which one to place
- **count:** how many of it. **0** means "place none, upgrade the ones earlier steps placed".
  For example, `Minigunner x0 up to 3` brings the first Minigunners up to level 3.
- **upgrade to:** the level to bring each one to (0 = don't upgrade)

How to edit:
- **Add a step:** pick the tower, count and level, then press **Add**. The step goes below the selected one.
- **Reorder:** use **Up** and **Down**.
- **Change a step:** select it, change the values, press **Update**.
- **Delete:** **Remove** deletes one step, **Clear** deletes the whole plan.
- **Save:** press **Save**. The bot uses the plan the next time it starts.

**How the bot follows it:**
1. **Top to bottom.** It works on the first step that isn't finished.
2. **One tower at a time.** Within a step it places one tower, upgrades it to the step's level (or until "Fully
   Upgraded"), then places the next. While doing a step it buys nothing else and saves up when cash is short.
3. **Tower not unlocked yet.** If the step's tower can't be placed until a later wave (its `from_wave`, e.g. Crook
   Boss at 10), the bot upgrades the towers it already has while it waits.
4. **Step impossible.** A step whose tower can't be placed at all is skipped: its `limit` is reached, there is no
   free spot, or its waves are over.
   **Step stuck.** If a step's tower fails 3 times in a row (the game keeps refusing it, or a Ranger finds no
   cliff), the bot sets that step aside for 2 minutes and goes on with the next one, so the defense keeps
   growing. The log says `plan step N skipped for 2 minutes (its tower keeps failing)`.
5. **After the last step.** Choose in the window:
   - **"AI keeps playing"** (`after_plan: "ai"`): the AI chooses freely within the tower limits and waves.
   - **"Only upgrade"** (`after_plan: "upgrade"`): the bot just keeps upgrading.

A planned tower is enabled in `actions` automatically when you save. The current step shows in the log and the
console, e.g. `plan 3/6: place Crook Boss 1 of 1 ($2,250)` or `plan 3/6: upgrading Crook Boss at spot 9 to
level 3 (level 1, next upgrade $1,800)`.

The default plan, for solo Molten (see [Molten notes](#molten-notes) for the reasoning):

| # | tower | count | upgrade to | why |
|---|---|---|---|---|
| 1 | Soldier | 3 | 1 | cheap defense for the first waves |
| 2 | Farm | 3 | 1 | cash early |
| 3 | Soldier | existing | 2 | sees hidden enemies: Molten Demons (hidden) come from wave 11 |
| 4 | Soldier | existing | 3 | 20 DPS each, range 19, before wave 11 |
| 5 | Crook Boss | 1 | 1 | $3,750 |
| 6 | Farm | existing | 2 | |
| 7 | Farm | 2 | 2 | five farms |
| 8 | Farm | existing | 3 | the best return in the game: $1,250 for +$300 a wave, paid back in about 4 waves |
| 9 | Crook Boss | 1 | 2 | second Crook Boss; level 2 sees hidden enemies |
| 10 | Crook Boss | existing | 2 | |
| 11 | Warden | 2 | 2 | melee at the path bends, best against crowds |
| 12 | Commander | 1 | 1 | $1,050: +15% fire rate for the top-right group |
| 13 | Crook Boss | existing | 3 | 156 DPS with goons, range 22: ready for the Molten Mech (23) and the Bulwark (25) |
| 14 | Warden | existing | 3 | |
| 15 | Farm | existing | 4 | +$400 a wave each, paid back in about 6 waves |
| 16 | Crook Boss | 1 | 3 | three |
| 17 | Warden | 2 | 3 | four Wardens |
| 18 | Commander | 1 | 1 | the bottom loop |
| 19 | Crook Boss | 1 | 3 | four |
| 20 | Warden | 2 | 3 | six Wardens, on three bends |
| 21 | Commander | existing | 3 | range 15, +17.5% |
| 22 | Commander | existing | 4 | +20%, range 17 |
| 23 | Warden | existing | 4 | level 4 is immune to stuns: the Warlord stuns every other tower |
| 24 | Crook Boss | 1 | 3 | five, the game's limit |
| 25 | Warden | 2 | 4 | eight Wardens |
| 26 | Warden | 4 | 4 | twelve, the game's limit |
| 27 | Commander | 2 | 4 | four Commanders: more fire rate and more Call to Arms / Support Caravan |
| 28 | Commander | 2 | 4 | six, the game's limit |
| 29 | Soldier | existing | 4 | |
| 30 | Soldier | 9 | 4 | fills the free tower slots up to the game's 40 |
| 31 | Crook Boss | existing | 4 | last: $47,500 for +222 DPS is the poorest buy |

**Why Crook Boss level 4 is last:** in the match that got the Warlord down to 15,000 health, the bot spent waves
32-34 saving for Crook Boss level 4 ($47,500) with $42,000-70,000 in hand, and only placed its Soldiers during
wave 35. A Commander or a level-4 Soldier gives more damage for the money, so they now come first.

### Second loadout: Minigunners late (`Start TDS Bot - Minigunner.bat`)

Advice from other players: late in the match, when cash piles up, place lots of Minigunners. They have no
placement limit, and a level-4 Minigunner does about 160 DPS, three times a level-4 Soldier in the same tower
slot. With the game's 40-tower cap, slots matter more than money at the end.

**`Start TDS Bot - Minigunner.bat`** starts the same bot with `config_minigunner.json`. Put this hotbar in the
lobby first:

| key | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| tower | Soldier | **Minigunner** | Crook Boss | Commander | Warden |

**What changes from the default plan:**
- **No Farms.** The two early Minigunners see hidden enemies from level 1.
- **Same core.** The Crook Bosses, the 12 Wardens and the 6 Commanders come in the same order.
- **The rest go to Minigunners.** After that, 28 towers are down and the remaining 12 tower slots get level-4
  Minigunners, one at a time ($31,700 each).

**What it costs:** no farm income, and that turned out to matter much more than estimated. First try
(2026-10-18 21:16): the bot played the plan without errors but lost at wave 30 (the Executioner). It had far fewer
towers at every wave than the Farm loadout's best match:

| wave | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|
| with Farms (20:15, reached 35) | 6 | 10 | 13 | 20 | 25 |
| Minigunner, no Farm (21:16) | 4 | 6 | 10 | 15 | 19 (lost) |

Five Farms at level 3-4 pay $2,500-4,500 a wave from about wave 14. The defense never caught up without that.
**Use `Start TDS Bot.bat` (with Farms)**; this launcher stays for experiments.

**How to compare:** run each launcher a few times. `matches.jsonl` records the wave each match reached.
`Edit Plan.bat` edits `config.json`. To edit the Minigunner plan, press **P** in the 3 s window when
`Start TDS Bot - Minigunner.bat` starts; the editor's title shows which file it is editing.

During a step that waits for its tower's wave (a tower with `from_wave`), the bot spends on upgrades. Move steps around in
`Edit Plan.bat` to change that.

The plan is stored in `config.json`, one step per line, so you can also edit it there:

```json
"plan": [
  {"tower": "Soldier", "count": 4, "upgrade_to": 2},
  {"tower": "Crook Boss", "count": 1, "upgrade_to": 3}
],
"after_plan": "ai"
```

Without a plan (an empty list), the AI decides everything, guided by each tower's `limit`, waves, `save_for` and
`upgrade_to`.

## 4. Start the brain and test it offline

Start the free Jev server and leave its window open:

```bat
start_onejev.bat
```

Then, in another terminal:

```bat
python tds_bot.py --test-brain calibration.png
```

This prints three decisions with latency and saves `ai_input_debug.jpg`, the exact image the AI sees.

- **`jev` brain:** first the number reader's line (`cash=1996 wave=9/30 towers=7/20`), then which actions are
  not offered and why, then each decision with its probability.
- **`ollama` brain:** each line shows the cash and wave it read.
- **Numbers come back `-1`?** Your regions are off: recalibrate them.

## 5. Run

```bat
python tds_bot.py --dry-run     # full pipeline, decisions logged, NO input sent
python tds_bot.py               # live
```

The bot **starts paused**. Click into Roblox, then press **Q** to start.

| option | effect |
|---|---|
| `--dry-run` | decide and log, never touch mouse or keyboard |
| `--backend jev\|ollama\|openai\|rules` | override `brain.backend` |
| `--model NAME` | override the model of the chosen backend (ollama / openai) |
| `--no-preview` | no OpenCV window |
| `--test-brain IMAGE` | run the AI on a screenshot and exit |
| `--probe-test [SLOT]` | in a match: hold a tower's preview over every spot without placing, save `probe_test.png` |
| `--autostart` | bring Roblox to the front and start playing after 3 s, without pressing Q (the one-click launcher uses this) |
| `--config FILE` | use another config file (e.g. one per map) |

Every decision is printed and appended to `bot_log.jsonl`. Each record also includes:
- `note`: events of that tick, e.g. "clicked READY", "... did not appear", "... is fully upgraded", "... forgotten"
- `bot_towers`: the towers the bot believes it has, so problems can be traced afterwards

## How the `jev` brain decides

A decision model answers questions with probabilities but can't write numbers. So the work is split:

1. **The number reader** (`reader.py`, `gemma3:4b` through Ollama) reads in its own thread, about every
   0.4-0.7 s:
   - cash
   - wave (`27/40`)
   - the tower counter (`12/20`)
   - the upgrade panel of a selected tower: its next price, or 0 for "Fully Upgraded!"

   Details:
   - **Cash:** a drop is believed at once, while a rise must show in two reads in a row that agree with each
     other, so a misread ($1,307 seen as $15,007) doesn't reach the bot. When the cash suddenly jumps up (more than
     kill money), the bot also reads the cash in the players list at the top right (`rois.cash_list`, white on
     grey, much easier to read) in a small separate request; when the two disagree the lower one counts. The model
     sometimes reads that list's shortened "$6.09K" as 609, so such a read is scaled by 10, 100 or 1,000 to
     whichever matches the bottom counter or the cash believed so far, and ignored if nothing matches. Tested on
     the current screen with eight painted values, two of them misread at the bottom: all eight right.
     (Asking for the list's cash inside the main request was tried first: the model then answered "no panel" for
     most upgrade prices, and the bot stopped upgrading. On a real upgrade panel with painted prices the main
     request reads 7 of 8 right; the miss is $47,500 read as "Fully Upgraded".) Money the bot spends is subtracted until the game's cash display shows it:
     the display lags a moment behind a purchase, and a reading taken in that moment still shows the old amount
     (a Minigunner was tried with $1,459 because a reading still said $4,209).
   - **Upgrade panel:** whether it is open is decided by looking for the red Sell button (a wide, solid red bar
     anywhere in `upgrade_panel`, read 0.8 s after the click so the panel has slid in), not by
     the AI.
   - **No reading yet, no action:** the bot does nothing until the reader has its first reading.
2. **The bot works out what is possible right now.** Rules are applied before the AI is asked:
   - placements: within each tower's `limit`, waves (`from_wave`/`until_wave`) and free spots, and only what the
     cash covers
   - build plan: while a plan step is running, only its tower (or upgrading the step's newest tower) is offered,
     and `WAIT` only when the cash doesn't cover it (see [Build plan](#build-plan))
   - focus (no plan): while a tower with `upgrade_to` is below that level, only upgrading it is offered (or `WAIT` until the
     cash covers its next upgrade)
   - saving (no plan): while a tower with `save_for` is due, only that tower is offered (or `WAIT` until the cash is there)
   - upgrade: only for towers the bot placed that are not fully upgraded and whose next upgrade (once read) the
     cash covers
   - sell: only once the bot has placed a tower
   - no new tower while the game is at its tower limit (`20/20`)
   - if a tower is affordable, `SKIP_WAVE` is not offered (spend before calling the next wave early)
   - if every offered tower is affordable, `WAIT` is not offered either (nothing to save for)
3. **Jev decides.** It gets the image plus the exact cash and wave, and answers in a single request:

   | question | type | used for |
   |---|---|---|
   | `in_match` | yes/no | Below 0.5 (lobby, menu, loading screen, another app): `WAIT`. |
   | `action` | choice over the possible actions | Probability per action. The top one is used if it reaches `brain.jev.min_confidence` (default 0.25), otherwise `WAIT`. |

4. **Placement.** The bot finds a valid position by looking at the screen (next section). Afterwards the tower
   counter must go up within 3 s. If it doesn't, the spot is freed again. After 2 failures that spot is skipped
   for the rest of the run.

The log shows the real numbers, e.g. `PLACE_TOWER_1 ... cash=18386 wave=27  p=0.45, next PLACE_TOWER_3 0.38`.

**Without the reader** (`reader.enabled: false`), Jev estimates cash itself with one yes/no question per tower
price ("the CASH panel shows at least $1,850"). The bot then picks the cash level that fits those answers best.
This is less exact: right 5 of 7 times on test frames.

**Tested on frames from a real match (RTX 5070):**
- **Reader:** read the real frame exactly ($1,996, wave 9/30, towers 7/20). On 8 frames with other numbers painted
  in, from $0 to $99,999 and waves 1-40, all 8 were read correctly.
- **Decisions:**

  | frame | choice |
  |---|---|
  | wave 3, $650 | Soldier (p=0.73) |
  | wave 9, $1,996 | Soldier (0.53) |
  | wave 27, $18,386 | Minigunner (0.45), then Shotgunner |
  | wave 34, $28,680 (live) | Minigunner |

## Placing towers

A calibrated spot is a starting point, not a fixed click. While you hold a tower, TDS shades every place you
can't put it in red: the path, scenery, a ring around each tower, and (for a Ranger) all ground that isn't a
cliff. The bot reads that shading:

1. **Closes the upgrade panel.** If a tower's panel is open (its red Sell button shows), the bot clicks empty
   ground to close it, because it covers the spots in the lower middle of the screen. It clicks the free spot or
   map point farthest from every tower, outside the HUD, and checks the Sell button is gone (up to 3 tries).
2. **Takes three pictures.** One with the mouse at the left edge of the screen and no tower held, one
   after pressing the tower's key (`placement.overlay_wait_s`, 0.4 s later), and one more with the held tower
   moved to the right edge (`placement.park_wait_s`, 0.15 s).
3. **Builds a no-go map.** A pixel is no-go if it turned redder or clearly changed in **both** held pictures.
   The held tower's own range circle (yellow, with a dashed red edge) and its Place / Rotate / Cancel menu follow
   the mouse, so they show up in only one of the two and drop out. Small areas closed in by red, such as the
   inside of a tower's ring, count as no-go too.
4. **Picks a point.** It shrinks the clear area by the tower's size (`placement.footprint_px`, 28 px; per tower
   with `footprint_px` in `towers`) and takes the clear point nearest the spot, at most
   `placement.max_shift_px` (150 px) away. The HUD, the screen edges and (if still open) the upgrade panel are
   never used.
5. **Clicks, then empties its hand.** After the click it moves the mouse back to the right edge and waits
   `placement.confirm_wait_s` (0.6 s) for the shading to fade. Then it checks whether the no-go areas look like
   the "holding" picture or the "free" one. Either way it presses **Q** (TDS's Cancel), which does nothing when
   no tower is held. So a tower can never stay in hand and get dropped by a later click.
   (Looking right at the click point was what went wrong before: the new tower shows its range while the mouse
   is on it, and the shading takes a moment to fade, so the bot thought the tower was refused and pressed its
   key again, which picked up another tower.)
6. **Confirms with the counter.** The tower counter must then go up within 3 s (see the previous section). With
   the number reader on, the counter has the last word; the picture check only decides without it.
7. **Grows new spots.** When all spots are used, it adds new ones around the towers it really placed, about
   70 px apart.

If the bot sees no red shading at all (less than `placement.min_overlay` of the screen changed), it falls back to
the older way: it slides the preview over rings of positions around the spot (`placement.search_px`) and clicks
the first one where the preview isn't red.

**Cliff-only towers (Ranger, `"cliff": true`)** don't use the calibrated spots, which are on the ground. With a
Ranger in hand, TDS shades everything except the cliffs, so the clear part of the map is the cliffs:
1. **Where it looks.** The clear point nearest its other Rangers, else nearest the middle of the map. Cliff
   points the game refused before are skipped.
2. **Placing.** It clicks there, and the point is kept for Rangers only.
3. **No cliff.** If the whole map is shaded, this map has no cliff it can use. The bot puts the Ranger back,
   says `no cliff for Ranger on this map`, and doesn't try again for 5 minutes. Meanwhile it spends on other
   towers (see [Build plan](#build-plan), rule 4).

**Debugging:** the first 12 placements of each run save a picture to `probe_debug/map_*.png`: the game with the
no-go map in red, a cross on the wanted spot and a green circle on the chosen point.

To check a map without placing anything, start a match and run:

```bat
python tds_bot.py --probe-test 2
```

This uses the old sweep: it holds tower 2's preview over every spot and its nearest ring, then puts it back. It
saves `probe_test.png`, with each tried position marked: green = ok, red = blocked, grey = hidden.

The shading map was tested on screenshots of the real game's shading and in a simulated map, not yet in a live
match. If placements go wrong, the `probe_debug/map_*.png` pictures show what the bot saw.

## Molten notes

Gathered from the official wiki and community guides (October 2026); waves and numbers change with updates.

**Turn off Auto Skip in the TDS settings for Molten.** A wave can be skipped after a third of its time, so with
Auto Skip on the next wave starts early and up to three waves pile onto the defense at once. Farms pay per wave,
not per minute, so turning it off costs no money, only time. The bot's runs were taking about 25 s a wave.

**Why Crook Boss is back (October 2026):** a leaked enemy hits the base for its remaining health, so the Bulwark
(wave 25, 6,000+ health, can't be stunned) leaking with a few thousand left ends the match even at 103/115 base
health. What kills it is damage times the time it spends in range as it walks by. Per $1,000 at level 3, a Crook
Boss (range 22, plus goons on the path) deals roughly 305 to a passing boss, a Minigunner (range 18) about 247 and a
Warden (melee, 7.5) about 195; Wardens shine against crowds, where something is always in range. Every run with
Crook Bosses passed wave 25 (three reached 30); the three runs with Minigunners, Wardens and Commanders but no
Crook Boss all died at 25.

**Wardens, first try (October 2026):** two matches lost at wave 25 to the boss there. The plan had spent
$35,000 on Farm levels 4 and 5 at waves 17-22 (level 5 hadn't paid back by wave 25) and was saving for Warden
level 4, so it had about 950 DPS at wave 25 against about 1,400 in the wave-30 runs. Farm level 5 is gone from the
plan; six Wardens at level 3 come first.

**Wave 28 run (October 2026):** with the new plan the bot reached wave 28. From wave 22 on it was saving for
Minigunner level 4 ($21,500 each) with up to $11,000 idle, so the plan now places more Crook Bosses and
Minigunners at level 3 first. Of six failed placements, one came from a cash misread (the players list showed
$1.21K, the bot read $2,616), one landed on the platform in the middle of the map and two on the barrels by the
red truck; neither is shaded red.

**What the bot's runs showed (October 2026):** every run with the old plan died at waves 21-23, with $10,000-
$18,000 unspent at waves 20-22. The plan was saving for a single $14,000 Crook Boss upgrade while the Molten Mech
(23) arrived. The current plan buys damage in smaller steps first, and puts more into Farms early.

**Damage per $1,000 spent** (Regular mode, tds.wiki):

| tower and level | total cost | DPS | DPS per $1,000 | range |
|---|---|---|---|---|
| Minigunner 2 | $3,700 | 25 | 6.8 | 18 |
| Minigunner 3 | $10,200 | 70 | 6.9 | 18 |
| Minigunner 4 | $31,700 | 160 | 5.0 | 21 |
| Crook Boss 2 (with 2 goons) | $6,500 | 49 | 7.5 | 19 |
| Crook Boss 3 (with goons) | $20,500 | 156 | 7.6 | 22 |
| Crook Boss 4 (with goons) | $68,000 | 378 | 5.6 | 22 |
| Warden 3 | $10,562 | 142 | 13.5 | 7.5 |
| Warden 4 (immune to stuns) | $27,062 | 296 | 10.9 | 8.5 |

Warden has twice the DPS per dollar but only melee range, so it must stand right at the path (`hug_path`);
spots 7, 8, 11 and 4 have the most path within its reach. In the loadout since the bot reached wave 30 twice.

**The Molten Warlord (wave 35):** 150,000 health solo, very slow. At 35% it turns invulnerable for a few seconds,
then heals back to full and calls 8 more enemies, so about 250,000 damage is needed in all. Its slams and Molten
Rain stun towers for 4.5-5.5 s; a fireball cuts a tower's damage by 60% for 10 s. Every 1% of its health lost
pays $2,500, money the bot keeps spending during the fight. Wave 34 brings four Molten Titans, and wave 35 adds
three Titans, an Executioner, three Bulwarks and a Tanker. Wardens at level 4 keep hitting through the stuns, and
the wiki recommends Wardens for this boss because it walks so slowly ([tds.wiki: Molten Warlord](https://tds.wiki/w/Molten_Warlord)).

**Farms** (income per wave): $60, $100 at level 1, $200 at 2, $500 at 3 (+$1,250), $900 at 4 (+$2,500), $1,500 at 5
(+$4,500). Level 3 pays back in about 4 waves, level 4 in 6, level 5 in 7.5.

- **Hidden enemies:** Molten Demons are hidden since v1.84.0 (January 2026) and come from **wave 11** (160 health, fast); Hidden Boss at wave 18, hidden Molten Necromancers from wave 20, more Hidden Bosses at 23-24. A leaked enemy hits the base for its remaining health, so one Demon getting through can end the match. Hidden detection: Soldier level 2, Warden level 1, Crook Boss level 2. Have hidden detection before
  then: Minigunner level 1, Crook Boss level 2. Rangers have none.
- **Bosses:**

  | wave | boss |
  |---|---|
  | 7 | Elite Abnormal |
  | 16 | Molten Golem |
  | 23 | Molten Mech (fast, 1,800 HP, 40% defense) |
  | 25 | Bulwark (6,000 HP, stun-immune) |
  | 30 | Executioner |
  | 32 | Summoner |
  | 34 | Titan |
  | 35 | **Molten Warlord** |

- **Warlord:** slams stun towers for 4.5-5.5 s, War Cry heals it, and it turns invulnerable for a while at 35%
  HP. Rangers are stun-immune; spread towers so one slam doesn't stop everything.
- **Owned but not in the loadout:**
  - **Turret:** sees hidden enemies from level 0, but costs $7,750 plus $131,250 in upgrades; too expensive solo.
  - **Electroshocker, Freezer:** slow fast enemies like the Mech; options if the Mech keeps breaking through.

Sources:
- [tds.wiki: Molten Mode](https://tds.wiki/w/Molten_Mode)
- [Ranger](https://tds.wiki/w/Ranger)
- [Turret](https://tds.wiki/w/Turret)
- [Minigunner](https://tds.wiki/w/Minigunner)
- [Crook Boss](https://tds.wiki/w/Crook_Boss)
- [Farm](https://tds.wiki/w/Farm)
- [Warden](https://tds.wiki/w/Warden)
- [Solo Molten farming guide](https://towerdefensesimulatorroblox.wiki/progression/solo-molten-coin-farming/)
- [Molten Mech](https://tds.wiki/w/Molten_Mech)
- [Bulwark](https://tds.wiki/w/Bulwark)
- [Molten guide (Warlord)](https://towerdefensesimulatorroblox.wiki/modes/molten/)
- [Destructoid tier list](https://www.destructoid.com/tower-defense-simulator-tower-tier-list/)

## Tower abilities

Commander abilities have keys: **F = Call to Arms** (from level 2: towers in range fire up to 20% faster for 10 s,
30 s cooldown, free) and **B = Support Caravan** (level 4: two gun trucks and a missile truck, $2,000, 60 s
cooldown). They show next to the open upgrade panel and in a bar on the left when nothing is selected. The bot
presses them itself (`abilities` in `config.json`):

- Call to Arms from wave 8, as soon as one is ready. The cooldown is shared out over the Commanders that have it,
  so with six it presses F every ~5 s and their buffs follow each other.
- Support Caravan while a boss health bar is up (the purple bar under the base health), and from wave 25 on
  whenever it is ready, if the cash covers it with $3,000 to spare. The plan takes Commanders to level 4 right
  after level 3, so the Caravan is there from about wave 27.

**Faster upgrades.** With a tower's `upgrade_costs` known, the bot sets the next price itself after pressing E and
can press again on the next tick instead of waiting about a second for the reader; a Warden goes from level 0 to 4
in four presses. The reader still checks the panel afterwards and corrects the level if a press didn't take.

| field | meaning |
|---|---|
| `name`, `key`, `tower`, `min_level` | the ability, its key, and the tower and level that have it |
| `cooldown`, `per_tower` | seconds between uses; with `per_tower` the cooldown is divided by how many towers have it |
| `from_wave`, `cost`, `keep` | not before this wave; its price, and how much cash to keep after paying it |
| `when: "boss"`, `always_from_wave` | only while a boss bar is up, or from this wave on anyway |

**Upgrade prices on a green box.** When an upgrade is affordable, TDS turns the price box green, price text
included, and the vision model then often reads it as "Fully Upgraded!" (a level-0 Commander's $400 was read as 0,
so the bot gave up on all Commanders until wave 34). The bot now looks at the level bars under the price: all lit
bright green means fully upgraded; otherwise a 0 reading is replaced by the known price for the tower's level
(`upgrade_costs`). Tested on five Commander panels, levels 0-4: all right.

## The camera during boss skills

A boss skill shakes the camera, and in the final fight the Warlord slowly turns the view (to the right) while the
light goes to sunset. Map clicks follow: the bot matches corner points (ORB) of a small grey copy of the screen
against the first frame of the match, leaving out the HUD, the ability buttons and the upgrade panel, and fits a
turn + zoom + shift (RANSAC). Buttons such as READY and Restart Match stay where they are. A fit needs at least 40
matching points; an unsure one is ignored. Only a sure change of more than 6 px between two fits, measured on the
matched map points, holds map clicks back for a second (keys like F and B never wait), and for at most 3 s in a
row: a shake is over by then, while a steady turn goes on and the clicks simply follow it.

**Why the 3 s limit:** in the match that left the Warlord at 15,000 health, the sunset turn kept the bot from
clicking for 75 s (20:37-20:38), while its cash went from $76,000 to $346,000. Back then movement was measured at
the far screen corners, where even a slow turn looks like more than 6 px. In a test with a 30 s turn, the old
check held the clicks for 15-24 s and the new one for 0-4 s; clicks land within 1.3 px.

Tested on screenshots turned 12-25 degrees, zoomed 0.9-1.05 and tinted like the sunset: every spot within 1 px,
about 40 ms a reading. (The first version only measured shifts, and held clicks back 641 times in one match on
unsure readings.) On the real lost-at-35 screenshot, with the result panel over the middle, too few points matched
to trust, so it would be ignored rather than acted on.

**The end-game money.** The game limits Warden to 12, Crook Boss to 5 and Commander to 6, and the bot reached all
three by wave 35 with up to $392,000 left and 9 of 40 tower slots free. The Soldier limit of 4 was the bot's own
setting, not the game's; it is gone, and the plan ends with nine more Soldiers to level 4 ($9,050 each, 54.5 DPS,
sees hidden and flying enemies).

`matches.jsonl` now says `"won": true` only when the result screen's title isn't the red YOU LOST. (Before, reaching
wave 35 counted as a win: the first wave-35 match, lost to the Warlord, was logged as won.)

## Endgame (waves 34-35): what the bot does

**Wave 35 is lost on purpose (`"final_wave": "lose"`, the default).** A win sends the game back to the lobby,
where the bot can't go on: after the 03:06 win on 2026-10-19 it sat there until 08:43. A loss at wave 35 still
pays coins and XP (550 coins at 35) and shows Restart Match, so the loop goes on. At the start of wave 35 the bot:
- sells every tower, Wardens first, then Crook Bosses and Commanders;
- stops upgrades and abilities.

The base falls and the next match starts, a few minutes sooner than playing the boss out. Other settings:
- `"final_wave": "idle"`: just stops. With level-4 Crook Bosses the towers may still win.
- `"final_wave": "win"`: plays the boss wave to win as below (Strongest, leapfrog Wardens). Use it only while
  watching, because a win ends in the lobby.

**The wave number is checked before the bot acts on it.** One bad reading (898, 797 or 880 around wave 20, with
cash read as 5) started the endgame at wave 20 in 4 of 10 overnight matches. Each time it sold the Farms, stopped
building, and lost at wave 28-30 with 8 towers. Now:
- a wave number over 60 or over the total is ignored;
- the next wave counts after 2 separate readings agree;
- any other jump needs 5;
- the same reading seen again in the next round counts once.

Replaying those 4 matches from `bot_log.jsonl`, the endgame no longer starts early. In normal matches it still
starts at wave 34 at the same moment, and a new wave is taken in 1.1 s (median).

From wave 34 (`endgame.from_wave`) the bot stops following the build plan and runs `endgame.py`:

1. **Sell** the Farms (income no longer matters) and any tower other than Warden, Crook Boss or Commander that
   stands more than 180 px from the path (out of range). In the winning game that was the 5 Farms and the 2
   farthest Soldiers.
2. **Top levels:** Crook Bosses to level 4 first, then any Warden or Commander not at its top level.
3. **Hold the cash** until the boss comes.
4. **Wave 35** (`"final_wave": "win"` only), boss bar up: every Crook Boss gets two clicks on its target arrow,
   First -> Last -> **Strongest**.
5. **Leapfrog Wardens** (`"win"` only) once the boss is 0.15 along the path:
   - sell a Warden the boss has walked more than 0.05 past;
   - place a new one 0.08-0.20 ahead of it, where the path has fewest Wardens;
   - bring it straight to level 4 (stun immune);
   - then make the next move.

   It keeps 12 Wardens and $2,500 for Support Caravan. Each move costs about $17,000 net (a level-4 Warden costs
   $27,062; selling one gives about $10,300 back). The boss pays $2,500 per 1% of health lost, which covers it.
6. **Abilities** (F, B) keep firing as before.

**Where the boss is:** `path` in config is the enemy path on Farm Lands as 60 map points, cave (0) to base (1).
It was traced from the red shading the game shows while a tower is held. The Warlord is a big grey robot with
bright yellow-orange parts. The tracker takes the largest yellow-orange blob on the path, near where the boss
should be by now; yellow things that were there in wave 34 don't count. Between sightings it walks on at 0.007 of
the path per second, the speed measured in the recording, and it never walks back. On the winning recording it
was off by at most 0.05 of the path (mean 0.016); the largest miss is while the Warlord stops to heal.

**Settings** (`config.json`, `"endgame"`):

| setting | meaning | default |
|---|---|---|
| `enabled` | turn the endgame off (the plan plays on) | true |
| `from_wave` | the wave it starts | 34 |
| `boss_wave` | the boss wave | 35 |
| `final_wave` | the boss wave: `"lose"` sell everything (the loop goes on), `"idle"` stop, `"win"` play to win | `"lose"` |
| `sell` | towers always sold | `["Farm"]` |
| `sell_far_px` | other towers this far from the path are sold, except those in `keep` | 180 |
| `keep` | never sold for distance | `["Warden", "Crook Boss", "Commander"]` |
| `max_out` | brought to their top level, in this order | Crook Boss, Warden, Commander |
| `target` | clicks on the target arrow per tower | `{"Crook Boss": 2}` = Strongest |
| `target_arrow` | where the arrow is | `[0.454, 0.755]` |
| `leapfrog.tower` | the tower that is moved | Warden |
| `leapfrog.behind` | sold when the boss is this far past it | 0.05 |
| `leapfrog.ahead` | placed this far ahead of the boss | `[0.08, 0.2]` |
| `leapfrog.keep_cash` | cash kept back | 2500 |
| `leapfrog.start_after` | first move once the boss is this far along | 0.15 |
| `boss_speed` | path per second between sightings | 0.007 |

**To play waves 34-35 yourself again,** set `"handover": {"wave": 34}`. The handover comes first: while it is
your turn, the endgame does nothing.

**Tested offline only:** on the winning recording's pictures (dry run), and through the bot's real clicks and keys
on the simulated game. Not yet in the real game.

## Waves 34-35: you play, the bot records

The handover is **off** now (`"wave": 0`): the bot plays the end itself (see [Endgame](#endgame-waves-34-35-what-the-bot-does)).
With `"wave": 34`, the bot plays waves 1-33. When the wave counter shows 34 (`handover.wave`), it **hands the game to you** for the
rest of the match:
1. **It beeps** three times and the preview says `YOUR TURN`. It stops clicking, upgrading and using abilities,
   and empties its hand if it was holding a tower.
2. **You play** waves 34 and 35 in Roblox. While it is your turn, **Q** is the game's cancel key again, not the bot's
   pause. **ESC** still stops the bot.
3. **It records what you do:**
   - every mouse click, with where it is on the map (following the camera turn);
   - every game key: tower keys 1-0, E, Q, F, B, X, R, C, V, Z, G, T, H;
   - the cash, wave and tower count at that moment;
   - a picture at every click (the click circled in red) and every 3 s;
   - the bot's towers at the handover (`towers.json`).
4. **When the match ends,** the recording closes with the result. `matches.jsonl` gets `"player_played": true`,
   and the bot clicks Restart Match and plays the next match from wave 1 as usual.

**Nobody at the computer?** If nobody clicks or presses a game key in Roblox within 45 s of the handover, the bot
plays on itself, so overnight runs still finish their matches.

**Where it goes:** `recordings/<date-time>-wave34/`:
- `events.jsonl`: one line per click, key or picture;
- `frames/`: the pictures;
- `towers.json`: the bot's towers at the handover;
- `summary.json`: how long you played, cash at the start and the end, presses per key, the result.

About 10 MB per recording.

**What the bot learns from it:** the bot doesn't learn by itself from one game. Send the recording folder (or
just say a new one is there). The plan is to read your clicks, keys and pictures and turn what worked into rules
for the bot's own waves 34-35: which towers you sold, placed or upgraded and where, and when you used abilities.

**The first win (2026-10-18 23:26, `recordings/20261018-232625-wave34`):** the player took over at wave 34 with
37 towers and $86,000, and won with the base at 125/125. From the recording:
1. **Wave 34, first 30 s:**
   - spent the cash on Crook Boss level 4 ($47,500);
   - sold the five Farms (income no longer matters) and the Soldiers at the entrance;
   - used Support Caravan.
2. **Wave 35, Crook Boss targeting:** set to "Strongest" on two Crook Bosses. The target arrow goes First Enemy
   -> Last Enemy -> Strongest, so that's two clicks. The Warlord has by far the most health.
3. **Wave 35, leapfrogging Wardens:** sold the Wardens the Warlord had already walked past and placed new ones on
   the path ahead of it, each brought straight to level 4 (E pressed 4-5 times). Cash stayed at $150,000-340,000
   from the boss's $2,500 per 1% of health, so moving them cost nothing that mattered. Wardens are melee and stun
   immune at level 4, and the Warlord is slow, so they stay in contact with it the whole way.
4. **Abilities:** Call to Arms (F) and Support Caravan (B) pressed whenever they were ready.

**Privacy:** only the keys above are recorded, and only while Roblox is the focused window. Typing in other
windows (or a key like Enter for chat) is never logged.

**Settings** (`config.json`, `"handover"`):
- `wave`: the wave you take over, 34 (35 = only the final wave; 0 = never, the bot plays everything);
- `nobody_s`: seconds to wait for you before the bot plays on, 45;
- `record`: save the recording, true;
- `beep`: true;
- `frame_every_s`: seconds between timed pictures, 3.

## Playing unattended (overnight)

Start it as usual (`Start TDS Bot.bat`) and leave it. It keeps playing match after match:

- **Restart Match:** when a match ends, TDS shows its result screen (YOU LOST, or a win) with a green
  **Restart Match** button under a large dark panel. Once the bot has seen that screen twice in a row, it clicks
  the button. While that screen shows, and for 8 s after the click, it does nothing else. If the screen is still
  up 8 s later (the click didn't take; a new match normally loads within 3 s), it clicks again; the match is
  counted once. The new match starts with READY, which the bot presses as before. Turn it off with
  `"auto_restart": false`.
- **A fresh start each match:** towers, moved and added spots, failures, blocked points and plan progress are
  forgotten. (The first overnight run kept blocked points between matches. A placement made on the result screen
  "failed" after each restart and blocked a good spot, until after 26 matches no spot was left and the bot lost at
  wave 3 for two hours. Fixed: no actions on the result screen, a tower counter that drops never counts as a
  failure, and if blocked points cover every place near a spot they are ignored for that try.)
- **In a match by the numbers:** when the reader sees a wave running, the AI's own "is this a match?" answer no
  longer stops the bot (on this map it hovered just under the limit, idling the bot 30-40% of the time).
- **`matches.jsonl`:** one line per finished match: when it ended, the wave reached, whether it was won, minutes,
  towers. The console prints e.g. `match 3 over at wave 27 after 15 min; clicked Restart Match`.
- **Stays awake:** while running, the bot asks Windows not to sleep or turn the screen off (`safety.keep_awake`).
  The request ends when the bot stops.
- **Gets the focus back:** if Roblox has been out of focus for 60 s while the bot isn't paused (a popup took it),
  the bot brings Roblox to the front again (`safety.refocus_after_s`, `0` = off). Press Q to pause first if you
  want to use the PC.
- Only the newest 40 `probe_debug/fail_*.png` pictures are kept.

Tested on two real YOU LOST screenshots (button found) and 34 in-match frames (no false alarm). A win's screen
hasn't been seen yet; if it has no green Restart Match button, the bot just waits there.

**Only one bot at a time.** Two copies on one game fight over the mouse and keys (it happened once: one old copy
was still open behind the new one, and every match was lost by wave 11). The bot now holds a Windows-wide flag
while it runs; a second copy, or `Start TDS Bot.bat` started again, says another bot is running and stops.

## Safety and control

| what | effect |
|---|---|
| **Q** | pause / resume immediately; the current action stops before its next key or click |
| **ESC** | kill the bot |
| Mouse into the **top-left corner** | kill (pydirectinput failsafe) |
| Close the preview window, or Ctrl+C | kill |

- Hotkeys are **global**: they work while Roblox has focus. That also means pressing Q in any other app toggles pause.
- Q is also TDS's **Cancel** key. The bot presses it after every placement to make sure no tower stays in hand;
  those presses of its own don't pause it. Pressing Q yourself still pauses.
- Input is **only sent while Roblox is the focused window**. Alt-tabbing away aborts the current action.
- Every AI decision is vetted first:
  - placing needs a free calibrated spot and enough cash
  - upgrading and selling need a placed tower
  - each action type has a cooldown (`cooldowns`)
- A decision that arrives after you paused is thrown away.

The **preview** shows the live game with regions and placement spots drawn on it, plus the run state,
the brain and its latency, the last AI action and reason, the result, the cash/wave/tower counts,
and a thumbnail of what the AI sees.

## What each action does

| action | input |
|---|---|
| `PLACE_TOWER_N` | press `N`, read the red no-go shading, click the clear point nearest the next free spot; then wait for the tower counter to confirm it |
| `UPGRADE_SELECTED` | click the tower to open its panel and read the price, then press `e` on a later tick if the cash covers it. It keeps upgrading the selected tower until that is fully upgraded or too expensive, then moves on to the next. |
| `SELL` | select the current or most recent tower, press `x` |
| `SKIP_WAVE` | press `f`, or click `skip_wave_button` if calibrated |
| `WAIT` | nothing |

If clicking a tower twice opens no upgrade panel **and** the game's tower counter shows fewer towers than the bot
thinks it placed, that tower isn't really there, and the bot forgets it. The counter check matters: an earlier
version forgot real Crook Bosses whose panel it misread, then bought more until it hit the game's limit. If the
panel won't open 3 times while the counter says the tower exists, the bot leaves that tower alone for 30 s.

The bot only knows about towers **it placed during this run**:
- It can't upgrade or sell towers you placed yourself.
- Start it at the beginning of a match, and restart it for each new match.

## Config reference (`config.json`)

| key | meaning |
|---|---|
| `capture.window_title` / `auto_find_window` / `region` / `fps` | what to capture, and how often |
| `brain.backend` | `jev`, `ollama`, `openai` or `rules` |
| `brain.min_interval` | minimum seconds between decisions |
| `brain.image_width` / `include_full_frame` | size of each UI crop sent to the AI; whether to add a map overview |
| `brain.jev.*` | `url` (local OneJev `http://127.0.0.1:8000`), `model`, `api_key_env`, `timeout`, `min_confidence` |
| `brain.ollama.*` | `url`, `model`, `timeout`, `keep_alive`, `max_tokens`, `think` |
| `brain.openai.*` | `base_url`, `model`, `api_key_env`, `timeout`, `max_tokens`, `json_mode` |
| `reader.*` | number reader: `enabled`, `url` (Ollama), `model` (`gemma3:4b`), `timeout` (seconds per read once loaded), `load_timeout` (seconds to wait while the model loads, default 180), `keep_alive`, `interval` (seconds between reads). Not used with the `ollama` brain, which reads numbers itself. |
| `handover.*` | the wave the player takes over (`wave`, 34; 0 = never), `nobody_s` (45: the bot plays on if nobody does), `record`, `beep`, `frame_every_s`. See [Waves 34-35](#waves-34-35-you-play-the-bot-records). |
| `endgame.*`, `path` | the end of the match: `from_wave` 34, what to sell, top levels, Crook Boss targeting, leapfrog Wardens; `path` = the enemy path, cave to base. See [Endgame](#endgame-waves-34-35-what-the-bot-does). |
| `placement.*` | `probe` (look before clicking), `footprint_px` (tower size kept clear of red, 28), `max_shift_px` (how far a tower may move from its spot, 150), `overlay_wait_s` (wait after pressing the key before reading the shading, 0.4), `park_wait_s` (wait after moving the held tower to the second edge, 0.15), `confirm_wait_s` (wait after the click before checking the tower left the hand, 0.6), `min_overlay` (share of the screen that must change to count as shading, 0.005), `search_px` (ring radii for the fallback sweep), `patch_px`, `settle_s`, `max_probe_s`, `red_ratio`, `min_changed`, `auto_spots` (grow new spots around placed towers), `auto_spot_px`, `cliff_search_s` (time limit for a cliff tower's search), `debug_images`. Pixel values are for a 1920-wide window and scale with it. |
| `actions` | which actions the AI may pick |
| `towers` | per hotbar slot: `name`, `cost` (`0` = unknown, so the cash check is skipped), `limit`, `from_wave`, `until_wave`, `cliff`, `save_for`, `upgrade_to`, `upgrade_costs`, `max_level`, `note`, `spots` (see section 3) |
| `check_loadout` | read the hotbar prices at the start and match towers to keys (default `true`) |
| `plan` | the build plan: `[{"tower": name, "count": n, "upgrade_to": level}, ...]`, top to bottom (edit with `Edit Plan.bat`) |
| `after_plan` | `"ai"` (the AI plays on) or `"upgrade"` (only upgrades) once the plan is done |
| `strategy_hint` | plain-language strategy sent with every request; tune this |
| `rois` | `[x, y, w, h]` fractions for the UI crops: `cash`, `cash_list` (your cash in the players list, top right), `towers_count`, `wave`, `hotbar`, `upgrade_panel`, `upgrade_box` (the green next-upgrade box), `sell_button` (no longer used: the Sell button is now found anywhere in `upgrade_panel`), `ready_button` (the pre-game READY button) (set by `calibrate.py`) |
| `placement_spots` | `[x, y]` fractions where towers go (set by `calibrate.py`) |
| `skip_wave_button` | `[x, y]` to click instead of pressing the skip key, or `null` |
| `keys` | game keys for towers 1-5, upgrade, sell, skip, and `cancel` (TDS's Cancel while placing, `q`; `""` = put a tower back with its own key instead) |
| `cooldowns` | seconds between actions of the same type |
| `hotkeys` | pause and kill keys (letters, digits, `ESC`, `F1`-`F12`, `PAUSE`, `END`, ...) |
| `safety` | `start_paused`, `require_focus`, `dry_run` |
| `auto_ready` | click the green READY button before wave 1 (`rois.ready_button`). It clicks once, then waits 6 s before clicking again, so it doesn't un-ready. |
| `input` | `key_hold`, `click_hold`, `step_delay` in seconds; raise them if inputs get missed |
| `preview` | `enabled`, `width`, `show_ai_input` |

## The OneJev server

- `start_onejev.bat` runs `onejev_server.py`, a thin wrapper around `qev serve --multimodal`.
- It listens on `127.0.0.1:8000` only, so other machines can't reach it.
- **About 3-4 GB of VRAM** with the 0.8B model.
- **CUDA graphs:** upstream only uses them while 8 GiB of VRAM is free. A 12 GB card running Roblox never has that,
  so every decision ran without them (~1 s). The wrapper lowers that reserve to `QEV_GRAPH_RESERVE_GB` (default 1.5),
  which brought it to ~0.16 s. Raise it if you see CUDA out-of-memory errors.
- **Other models:** `start_onejev.bat --model onejev1.6-4b`. The 4B is about 10 GB in bf16, too big next to Roblox on
  a 12 GB card; `--quantize fp8` may make it fit. *Not tested here.*

**Hosted Jev (optional, costs money):** set `brain.jev.url` to `https://api.typesafe.ai` and
`set TYPESAFE_API_KEY=...`. The bot then sends the image as a TypeSafe asset with the
`TypeSafe-Danger-Zone-Images` header. Image input is experimental on TypeSafe's side.

## Troubleshooting

| problem | fix |
|---|---|
| `is the OneJev server running? Start start_onejev.bat` | Start it and wait for `Uvicorn running on http://127.0.0.1:8000`. |
| Log shows `cash=-1` / `[reader] ... is Ollama running?` | The number reader can't reach Ollama. Start Ollama (`ollama serve`) and check `ollama list` shows `gemma3:4b`. |
| `[reader] cannot reach ... timed out` over and over | Ollama runs but the model didn't load in time. Loading `gemma3:4b` cold from disk can take 30-60 s, and Ollama cancels a load when the bot stops waiting, so with a short wait it never finished. Fixed: the launcher loads the model while the Jev brain starts, and the reader waits up to `reader.load_timeout` (180 s) until the model has answered once. If it still happens, raise `load_timeout`. |
| Cash or wave read wrong | Check the `cash`, `towers_count` and `wave` crops in `ai_input_debug.jpg`. Each must hold its number fully, including room for more digits. |
| `... did not appear at spot N` | The tower counter didn't go up after a placement. Scenery such as barrels or trucks isn't shaded red, so the point looked clear. The next try at that spot picks a point at least 40 px away; after 2 failures the spot is skipped for that tower. Each failure saves `probe_debug/fail_*.png` with what the bot saw. |
| `... is level N (by its price), not M` | The bot's own count of upgrades was off (a press that didn't take, or a click that opened a neighbouring tower); it now uses the level the price shows. |
| `... the panel shows another tower's price ($N); trying again` | The click on a tower opened a different tower type's panel. The bot closes it and clicks again 3 s later. Towers on spots the open upgrade panel covers (Farm spots 14-16) get the panel closed first, since a click there lands on the panel. |
| `... clicking it keeps opening another tower; leaving it for 60 s` | Three wrong panels in a row for that tower: the plan goes on with other towers and comes back later. |
| `... looks fully upgraded (level N); checking again` | The upgrade panel showed no price below the tower's `max_level`, usually because the Commander's message box covered it. The bot looks again 5 s later. |
| `no clear place near spot N ({'mode': 'map', ...})` | Everything within `placement.max_shift_px` of the spot is shaded red. Move the spot, or look at `probe_debug/map_*.png`. |
| `no clear place near spot N ({'mode': 'sweep', ...})` | No red shading was seen, so the old sweep ran and found nothing. The tower key may not have picked up a tower (not enough cash, wrong `keys.tower_N`), or the shading takes longer to show: raise `placement.overlay_wait_s`. |
| `the game refused X near spot N` | Only without the number reader: the click didn't place the tower, so the bot put it back. After 2 refusals that spot is skipped for that tower. With the reader you see `did not appear at spot N` instead. |
| `Towers N in game` higher than `placed by the bot` | Something placed towers the bot doesn't know about. Before this fix: the bot thought a placed tower was refused and picked up another one. If it still happens, send `bot_log.jsonl` and the `probe_debug/map_*.png` pictures. |
| `no cliff for Ranger on this map` | With a Ranger held, the whole map was shaded red: no usable cliff. The bot retries every 5 minutes. If your map has none, remove Ranger from the plan and the hotbar. |
| It places in clearly wrong positions | Look at `probe_debug/map_*.png`. If parts that should be red are clear, the shading hadn't fully appeared yet: raise `placement.overlay_wait_s`. |
| It doesn't click READY | Check that `rois.ready_button` covers the green button on the pre-game screen (`python calibrate.py --only rois` while it shows). |
| It never upgrades (`opened ... upgrade panel` over and over) | The panel check doesn't see the Sell button. Check that `upgrade_panel` covers the whole panel including Sell, and that `upgrade_box` covers the green next-upgrade box. Recalibrate with a tower selected: `python calibrate.py --only rois`. |
| `jev` decisions take ~1 s instead of ~0.2 s | Server log says `CUDA graphs dropped ... GraphMemoryLow`. Free some VRAM or lower `QEV_GRAPH_RESERVE_GB`. |
| `jev` always answers `WAIT ... not in a match` | Check `ai_input_debug.jpg`. The crops must show the game, and Roblox must be on screen, not covered. |
| Keys or clicks ignored in Roblox | Roblox must be focused and on the primary monitor, at the same admin level as the terminal. Raise `input.step_delay` and `click_hold`. |
| Tower preview appears but isn't placed | The spot is invalid (on the path or too close). Recalibrate spots. |
| `Roblox window not found` in preview | Roblox isn't running or is minimized. The bot falls back to `capture.region`. |
| `ollama` brain: `used its whole token budget thinking` | Set `"think": false` under `brain.ollama`. |

## Files

| file | purpose |
|---|---|
| `tds_bot.py` | main loop: capture, preview, brain thread, CLI |
| `brain.py` | AI backends (jev, ollama, openai, rules), image composition, parsing |
| `reader.py` | number reader: cash, wave and tower counter via `gemma3:4b` |
| `placement.py` | look-before-you-click placement: red no-go map, clear-point picking, refusal check, fallback sweep, debug images |
| `controls.py` | Roblox window tracking, failsafe hotkeys, DirectInput executor |
| `calibrate.py` | interactive region, spot and skip-button calibration |
| `Start TDS Bot.bat`, `launch.py` | one-click start: Jev brain, Roblox check, spots, bot, clean-up |
| `Start TDS Bot - Minigunner.bat`, `config_minigunner.json` | the same with the Minigunner loadout and plan |
| `recorder.py`, `recordings/` | records how you play the waves the bot hands over (34-35) |
| `endgame.py` | waves 34-35 the way the player won: sell, top levels, Strongest, leapfrog Wardens |
| `Edit Plan.bat`, `plan_editor.py` | window to edit the build plan |
| `configio.py` | reads and writes `config.json` (keeps short lists on one line) |
| `setup.bat` | one-time install of both venvs (run automatically) |
| `onejev_server.py`, `start_onejev.bat` | the free local Jev server, if you want to start it by hand |
| `config.json` | all settings |

## Disclaimer

Automating gameplay may break Roblox's Terms of Use or the game's own rules, and can get an account
moderated. Use it at your own risk, ideally on an alt account in a private server.
