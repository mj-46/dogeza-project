"""学習した方策（.pt）で G1 を動かす。お手本（JSON）は使わず、方策と経過時間だけで動く。"""

import json
import re
from pathlib import Path

import numpy as np
import omni.physx
import torch

from dogeza.g1 import control, physics_player

POLICIES_DIR = Path(__file__).resolve().parents[3] / "assets" / "policies"
POSES_DIR = Path(__file__).resolve().parents[3] / "assets" / "poses"


def load_pose(name):
    """assets/poses/{name}.json を読む。{"pelvis_position", "pelvis_pitch", "joints": {関節名: 角度}}（control.apply_pose にそのまま渡せる）"""
    with open(POSES_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


# ---- 学習時（dogeza_rl/env_cfg.py）と同じ値にそろえる ----
DECIMATION = 4  # 物理 4 回に 1 回、方策が判断する（200Hz ÷ 4 = 50Hz）
POLICY_DT = 4 / 200  # 判断の間隔 [s]
ACTION_SCALE = 1.0  # 行動 1.0 = 土下座の最終姿勢（GOAL_JOINTS）から 1 rad
TIME_OBS_MAX_S = 12.0  # 観測の「経過時間」はこの秒数で 1 になり、その後は 1 のまま
STAND_POSE = load_pose("stand")  # 最初の姿勢であり、観測の関節角度の基準
GOAL_JOINTS = load_pose("dogeza")[
    "joints"
]  # 行動の基準（行動 0 = 土下座の最終姿勢を目標にする）


class PolicyPlayer:
    """PolicyPlayer(articulation)(t) で、学習した方策に G1 を動かさせる。"""

    def __init__(self, articulation, policy_file="dogeza_first_half.pt"):
        self.articulation = articulation
        self.policy = torch.jit.load(
            str(POLICIES_DIR / policy_file), map_location="cpu"
        )
        self.policy.eval()
        self.dof_names = articulation.dof_names
        self.default_joint_pos = np.array(
            [STAND_POSE["joints"].get(n, 0.0) for n in self.dof_names]
        )
        self.goal_joint_pos = np.array(
            [GOAL_JOINTS[n] for n in self.dof_names]
        )  # 行動の基準（土下座の最終姿勢）

        # 関節のモーターの強さ（学習時と同じ）
        stiffness = [
            next(k for pat, k, _ in physics_player.GAINS if re.fullmatch(pat, n))
            for n in self.dof_names
        ]
        damping = [
            next(d for pat, _, d in physics_player.GAINS if re.fullmatch(pat, n))
            for n in self.dof_names
        ]
        articulation.set_dof_gains(
            stiffnesses=np.array([stiffness]), dampings=np.array([damping])
        )

        control.apply_pose(articulation, STAND_POSE)
        self.actions = np.zeros(
            len(self.dof_names)
        )  # 学習のリセット直後と同じく、前回の行動は 0 から
        self.started = False  # 録画の準備中（t = 0）は、方策を使わずに立ち姿勢を保つ
        self.frame = 0  # 方策が判断した回数（経過時間の計算用）
        self.physics_steps = 0
        # 物理の 1 ステップごとに呼ばれる関数を登録（画面の更新とは別に、学習時と同じ 50Hz で判断するため）
        self.subscription = (
            omni.physx.get_physx_interface().subscribe_physics_on_step_events(
                self._on_physics_step, pre_step=True, order=0
            )
        )

    def __call__(self, t):
        if t > 0:
            self.started = True

    def _on_physics_step(self, dt):
        if self.physics_steps % DECIMATION == 0:
            if self.started:
                self._decide()
                self.frame += 1
            else:
                self.articulation.set_dof_position_targets(
                    self.default_joint_pos[None]
                )  # 開始前は立ったまま
        self.physics_steps += 1

    def _decide(self):
        """観測を作って方策に渡し、関節の目標角度を決める（学習時の _get_observations と同じ順番・計算）。"""
        positions, orientations = self.articulation.get_world_poses()
        _, angular = self.articulation.get_velocities()
        rotation = physics_player._quat_to_matrix(
            physics_player._np(orientations)[0]
        )  # (w, x, y, z)
        joint_pos = physics_player._np(self.articulation.get_dof_positions())[0]
        joint_vel = physics_player._np(self.articulation.get_dof_velocities())[0]
        obs = np.concatenate(
            [
                [
                    min(self.frame * POLICY_DT / TIME_OBS_MAX_S, 1.0)
                ],  # 1: 経過時間（0 〜 1）
                rotation.T @ np.array([0.0, 0.0, -1.0]),  # 3: 腰から見た重力の向き
                rotation.T
                @ physics_player._np(angular)[0],  # 3: 腰の角速度（腰から見た向き）
                [physics_player._np(positions)[0][2]],  # 1: 腰の高さ
                joint_pos - self.default_joint_pos,  # 29: 関節角度（立ち姿勢からの差）
                joint_vel,  # 29: 関節角速度
                self.actions,  # 29: 前回の行動
            ]
        )
        with torch.no_grad():
            self.actions = self.policy(torch.tensor(obs, dtype=torch.float32)[None])[
                0
            ].numpy()
        targets = (
            self.goal_joint_pos + ACTION_SCALE * self.actions
        )  # 学習時と同じく、土下座の最終姿勢が基準
        self.articulation.set_dof_position_targets(targets[None])
