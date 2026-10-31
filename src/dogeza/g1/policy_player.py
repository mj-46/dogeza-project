"""学習した方策（.pt）で G1 を動かす。お手本（JSON）は使わず、方策と経過時間だけで動く。
撮影用：「立位で静止 → 前半の方策（立つ → 土下座）→ 後半の方策（土下座 → 跪座）」を 1 テイクとし、
カメラを変えながら CAMERAS の数だけ繰り返す（テイクの始めに立ち姿勢へ戻す）。"""

import json
import re
from pathlib import Path

import numpy as np
import omni.physx
import torch

from dogeza.g1 import control, physics_player
from dogeza.scene import builder

POLICIES_DIR = Path(__file__).resolve().parents[3] / "assets" / "policies"
POSES_DIR = (
    Path(__file__).resolve().parents[3] / "assets" / "poses"
)  # 立ち姿勢・土下座・跪座の JSON


def load_pose(name):
    """assets/poses/{name}.json を読む。{"pelvis_position", "pelvis_pitch", "joints": {関節名: 角度}}（control.apply_pose にそのまま渡せる）"""
    with open(POSES_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


# ---- 撮影の段取り [s]（1 テイク = STAND_S + DOWN_S + UP_S 秒） ----
STAND_S = 3.0  # 立位で静止
DOWN_S = 10.0  # 前半の方策：立位 → 土下座
UP_S = 10.0  # 後半の方策：土下座 → 跪座
CAMERAS = [
    "/World/Cameras/FrontLeft",
    "/World/Cameras/Front",
    "/World/Cameras/Side",
]

# ---- 学習時と同じ値にそろえる ----
DECIMATION = 4  # 物理 4 回に 1 回、方策が判断する（200Hz ÷ 4 = 50Hz）
POLICY_DT = 4 / 200  # 判断の間隔 [s]
ACTION_SCALE = 1.0  # 前半：行動 1.0 = 土下座の最終姿勢から 1 rad
TIME_OBS_MAX_S = (
    12.0  # 前半：観測の「経過時間」はこの秒数で 1 になり、その後は 1 のまま
)
UP_ACTION_SCALE = 1.0  # 後半：脚・腰の行動の倍率（学習時の action_scale）
UP_ARM_ACTION_SCALE = 0.1  # 後半：腕の行動の倍率（学習時の arm_action_scale）
UP_TIME_OBS_MAX_S = 10.0  # 後半：観測の「経過時間」の上限（学習時の time_obs_max_s）
UP_RAMP_S = 3.0  # 後半：行動の基準を土下座から跪座へ移す秒数（学習時の ramp_s）

STAND_POSE = load_pose("stand")  # 最初の姿勢であり、観測の関節角度の基準
GOAL_JOINTS = load_pose("dogeza")[
    "joints"
]  # 前半の行動の基準（行動 0 = 土下座の最終姿勢を目標にする）
KIZA_JOINTS = load_pose("kiza")["joints"]  # 後半の行動の基準の行き先（跪座）

# 判断の回数に直した段取り
STAND_N, DOWN_N, UP_N = (round(s / POLICY_DT) for s in (STAND_S, DOWN_S, UP_S))


class PolicyPlayer:
    """PolicyPlayer(articulation)(t) で、学習した方策に G1 を動かさせる（t は main 側の経過時間。毎フレーム呼ぶ）。"""

    def __init__(
        self,
        articulation,
        policy_file="dogeza_first_half.pt",
        up_policy_file="dogeza_second_half.pt",
    ):
        self.articulation = articulation
        self.policies = [
            torch.jit.load(str(POLICIES_DIR / f), map_location="cpu").eval()
            for f in (policy_file, up_policy_file)
        ]
        self.dof_names = articulation.dof_names
        self.default_joint_pos = np.array(
            [STAND_POSE["joints"].get(n, 0.0) for n in self.dof_names]
        )
        self.goal_joint_pos = np.array(
            [GOAL_JOINTS[n] for n in self.dof_names]
        )  # 前半の行動の基準（土下座の最終姿勢）
        self.kiza_joint_pos = np.array(
            [KIZA_JOINTS[n] for n in self.dof_names]
        )  # 後半の行き先（跪座）
        arm = np.array(
            [
                any(w in n for w in ("shoulder", "elbow", "wrist"))
                for n in self.dof_names
            ]
        )
        self.up_scale = np.where(arm, UP_ARM_ACTION_SCALE, UP_ACTION_SCALE)

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

        self.take = 0  # 今のテイク（0 始まり）
        self.step = 0  # テイクの中で方策が判断した回数
        self.need_restart = True  # 次のフレームで、立ち姿勢に戻してカメラを切り替える
        self.started = False  # 録画の準備中（t = 0）は、立ち姿勢のまま待つ
        self.actions = np.zeros(len(self.dof_names))
        self.physics_steps = 0
        self._restart()
        # 物理の 1 ステップごとに呼ばれる関数を登録（画面の更新とは別に、学習時と同じ 50Hz で判断するため）
        self.subscription = (
            omni.physx.get_physx_interface().subscribe_physics_on_step_events(
                self._on_physics_step, pre_step=True, order=0
            )
        )

    def __call__(self, t):
        if t > 0:
            self.started = True
        if self.need_restart and self.take < len(CAMERAS):
            self._restart()
        elif (
            self.step < STAND_N
        ):  # 立位の静止中は、毎フレーム立ち姿勢に置き直す（目標角度だけでは自分で立っていられず前に倒れるため）
            control.apply_pose(self.articulation, STAND_POSE)

    def _restart(self):
        """テイクの始め：カメラを切り替え、G1 を立ち姿勢に戻す（物理の計算の外で行う）。"""
        builder.move_record_camera(CAMERAS[self.take])
        control.apply_pose(self.articulation, STAND_POSE)
        self.step = 0
        self.actions = np.zeros(len(self.dof_names))
        self.need_restart = False
        print(
            f"[dogeza] テイク {self.take + 1}/{len(CAMERAS)}（カメラ {CAMERAS[self.take]}）",
            flush=True,
        )

    def _on_physics_step(self, dt):
        if self.physics_steps % DECIMATION == 0:
            self._decide()
        self.physics_steps += 1

    def _decide(self):
        """段取りに合わせて、立ち姿勢を保つ・前半の方策・後半の方策のどれかで関節の目標角度を決める。"""
        k = self.step
        if self.need_restart or self.take >= len(CAMERAS):
            return  # テイクの切れ目・全テイクの後は、直前の目標角度のまま（跪座を保つ）
        if not self.started or k < STAND_N:
            self.articulation.set_dof_position_targets(
                self.default_joint_pos[None]
            )  # 立位で静止
        elif k < STAND_N + DOWN_N:
            frame = k - STAND_N  # 前半の方策を使い始めてからの判断の回数
            if frame == 0:
                self.actions = np.zeros(
                    len(self.dof_names)
                )  # 学習のリセット直後と同じく、前回の行動は 0 から
            self._act(0, frame, self.goal_joint_pos, ACTION_SCALE, TIME_OBS_MAX_S)
        else:
            frame = k - STAND_N - DOWN_N  # 後半の方策を使い始めてからの判断の回数
            if frame == 0:
                self.actions = np.zeros(len(self.dof_names))
            s = min(frame * POLICY_DT / UP_RAMP_S, 1.0)
            s = (
                s * s * (3 - 2 * s)
            )  # 学習時の DogezaUpEnv と同じ、始めと終わりがゆっくりの曲線
            base = self.goal_joint_pos + s * (self.kiza_joint_pos - self.goal_joint_pos)
            self._act(1, frame, base, self.up_scale, UP_TIME_OBS_MAX_S)
        if self.started:
            self.step += 1
            if self.step >= STAND_N + DOWN_N + UP_N:  # テイクの終わり → 次のテイクへ
                self.take += 1
                self.need_restart = True

    def _act(self, half, frame, base, scale, time_obs_max_s):
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
                [min(frame * POLICY_DT / time_obs_max_s, 1.0)],  # 1: 経過時間（0 〜 1）
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
            self.actions = self.policies[half](
                torch.tensor(obs, dtype=torch.float32)[None]
            )[0].numpy()
        self.articulation.set_dof_position_targets((base + scale * self.actions)[None])
