import math
import re

import numpy as np
import omni.timeline

from dogeza.g1 import control, keyframes

# 関節のモーターの強さ（剛性 [N·m/rad], 減衰 [N·m·s/rad]）。上から順に、最初に一致したものを使う
GAINS = [
    (r".*_hip_.*", 150.0, 5.0),
    (r".*_knee_joint", 200.0, 6.0),
    (r".*_ankle_.*", 40.0, 2.0),
    (r"waist_.*", 200.0, 5.0),
    (r".*_(shoulder_.*|elbow_joint)", 50.0, 2.0),
    (r".*_wrist_.*", 20.0, 1.0),
]
KP_POSITION = 10.0  # 腰の位置の補助の強さ [1/s]
KP_ROTATION = 10.0  # 腰の向きの補助の強さ [1/s]


def _np(x):
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=float)


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_error(r_target, r_current):
    """今の向きから目標の向きへの回転を、回転ベクトル（軸 × 角度 [rad]）で返す。"""
    r = r_target @ r_current.T
    v = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    s = math.sin(angle)
    return v if s < 1e-6 else v * (angle / s)


class PhysicsPlayer:
    """PhysicsPlayer(articulation, motion, root_assist)(t) で、時刻 t の姿勢を目標に物理で動かす。

    root_assist: 腰を目標の位置・向きへ引き寄せる補助の割合（0 = 補助なしの純粋な物理, 1 = 強く補助）。
                 関節の動きや床との接触は、どちらの場合も物理で計算される。
    """

    def __init__(self, articulation, motion, root_assist=1.0):
        self.articulation = articulation
        self.motion = motion
        self.root_assist = root_assist
        self.timeline = omni.timeline.get_timeline_interface()
        self.last_report = -1
        self.dof_names = (
            articulation.dof_names
        )  # 関節名の並び順（毎コマ使うので最初に 1 回だけ取得）

        # 関節のモーターの強さを設定
        stiffness = [
            next(k for pat, k, _ in GAINS if re.fullmatch(pat, n))
            for n in self.dof_names
        ]
        damping = [
            next(d for pat, _, d in GAINS if re.fullmatch(pat, n))
            for n in self.dof_names
        ]
        articulation.set_dof_gains(
            stiffnesses=np.array([stiffness]), dampings=np.array([damping])
        )

        # 開始時に 1 回だけ、最初の姿勢に置き直す
        control.apply_pose(articulation, keyframes.pose_at(motion, 0.0))
        self.sim_start = self.timeline.get_current_time()

    def __call__(self, t):
        pose = keyframes.pose_at(self.motion, t)

        # 関節：目標角度だけをモーターに渡す（実際の動きは物理が計算する）
        targets = np.array([[pose["joints"].get(n, 0.0) for n in self.dof_names]])
        self.articulation.set_dof_position_targets(targets)

        # 腰：目標の位置・向きとのずれに応じて、速度で補助する
        if self.root_assist > 0:
            positions, orientations = self.articulation.get_world_poses()
            linear, angular = self.articulation.get_velocities()
            p, q = _np(positions)[0], _np(orientations)[0]
            pitch = pose["pelvis_pitch"]
            r_target = np.array(
                [
                    [math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)],
                ]
            )
            v_cmd = KP_POSITION * (np.array(pose["pelvis_position"]) - p)
            w_cmd = KP_ROTATION * _rotation_error(r_target, _quat_to_matrix(q))
            a = self.root_assist
            v = (1 - a) * _np(linear)[0] + a * v_cmd
            w = (1 - a) * _np(angular)[0] + a * w_cmd
            self.articulation.set_velocities(v[None], w[None])

        # 1 秒ごとに、時刻のずれと腰の高さを表示（物理の時間が t と一致しているかの確認用）
        if int(t) != self.last_report:
            self.last_report = int(t)
            z = _np(self.articulation.get_world_poses()[0])[0][2]
            sim_t = self.timeline.get_current_time() - self.sim_start
            print(
                f"[dogeza] t={t:5.1f}s  物理の時刻={sim_t:5.1f}s  腰の高さ={z:.2f}m（目標 {pose['pelvis_position'][2]:.2f}m）"
            )
