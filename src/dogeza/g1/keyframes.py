import json
from pathlib import Path

import numpy as np

KEYFRAMES_DIR = Path(__file__).resolve().parents[3] / "assets" / "keyframes"


def load_motion(filename):
    with open(KEYFRAMES_DIR / filename, encoding="utf-8") as f:
        data = json.load(f)
    frames = data["keyframes"]
    return {
        "joint_names": data["joint_names"],
        "times": np.array([k["time"] for k in frames]),
        "pelvis_positions": np.array([k["pelvis_position"] for k in frames]),
        "pelvis_pitches": np.array([k["pelvis_pitch"] for k in frames]),
        "joints": np.array([k["joints"] for k in frames]),
    }


def pose_at(motion, t):
    """時刻 t [秒] の姿勢を、前後のキーフレームから直線的に補間して求める。"""
    times = motion["times"]
    t = min(max(t, times[0]), times[-1])  # 範囲外は最初・最後の姿勢のまま
    position = [np.interp(t, times, motion["pelvis_positions"][:, i]) for i in range(3)]
    pitch = np.interp(t, times, motion["pelvis_pitches"])
    angles = [
        np.interp(t, times, motion["joints"][:, i])
        for i in range(len(motion["joint_names"]))
    ]
    return {
        "pelvis_position": position,
        "pelvis_pitch": float(pitch),
        "joints": dict(zip(motion["joint_names"], angles)),
    }
