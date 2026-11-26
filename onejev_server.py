"""Start the free local Jev-compatible server (OneJev) for the bot's "jev" brain.

Same as `qev serve --multimodal`, tuned for a consumer GPU: upstream only uses CUDA graphs
while 8 GiB of VRAM is free, which a 12 GB card running Roblox never has, so every decision
ran eagerly. Here the reserve is QEV_GRAPH_RESERVE_GB (default 1.5). Listens on 127.0.0.1 only.
Run it with the OneJev venv (start_onejev.bat does this):

  .venv-onejev\\Scripts\\python onejev_server.py --model onejev1.6-0.8b
"""
import functools
import os
import sys

import qev.mm_engine as mm_engine
from qev.cli import main

reserve = int(float(os.environ.get("QEV_GRAPH_RESERVE_GB", "1.5")) * (1 << 30))
mm_engine._make_room = functools.partial(mm_engine._make_room, reserve_fraction=0.0, reserve_bytes=reserve)

if __name__ == "__main__":
    args = sys.argv[1:]
    for flag, value in (("--model", "onejev1.6-0.8b"), ("--host", "127.0.0.1"), ("--port", "8000")):
        if flag not in args:
            args += [flag, value]
    sys.argv = ["qev", "serve", "--multimodal", *args]
    main()
