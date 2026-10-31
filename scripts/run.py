"""isaac sim_再生用"""

import argparse
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--record", action="store_true")
args = parser.parse_args()

# isaac simの起動
simulation_app = SimulationApp({"headless": False})


try:
    from dogeza import main

    main.run(simulation_app, args.duration, args.record)
except Exception:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
