import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

from isaaclab_rl.entrypoints import run_train_cli

import dogeza_rl  # noqa: F401  前半 "dogeza-first-half" と後半 "dogeza-second-half" を Isaac Lab に登録する

# どちらを学習するかは、必ず --task で指定する
if "--task" not in sys.argv:
    sys.exit(
        "--task dogeza-first-half（前半）か --task dogeza-second-half（後半）を指定してください"
    )

# Isaac Lab 付属の学習処理（RSL-RL の PPO）に、指定したタスクを渡して実行する
sys.exit(run_train_cli(["--rl_library", "rsl_rl", *sys.argv[1:]]))
