import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)  # 学習結果を dogeza-project/logs/ から探す

from isaaclab_rl.entrypoints import run_play_cli

import dogeza_rl  # noqa: F401  土下座タスク "Dogeza-G1" を Isaac Lab に登録する（F401はruffで消えないように）

# Isaac Lab 付属の再生処理に、土下座タスクを渡して実行する
sys.exit(
    run_play_cli(
        ["--rl_library", "rsl_rl", "--task", "Dogeza-G1", "--viz", "kit", *sys.argv[1:]]
    )
)
