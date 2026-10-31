"""後半（土下座 → 跪座）の学習環境の本体。前半の環境（first_half/env.py）を引き継ぎ、「行動の基準と倍率」と「報酬」だけを変える。"""

import math

import torch

from dogeza_rl.first_half.env import DogezaEnv
from dogeza_rl.second_half.env_cfg import KIZA_JOINTS, DogezaUpEnvCfg


class DogezaUpEnv(DogezaEnv):
    """後半：土下座 → 跪座。前半と違うのは「行動の基準と倍率」と「報酬」だけ（観測・終了条件・開始のしかたは前半と同じ）。"""

    cfg: DogezaUpEnvCfg

    def __init__(self, cfg: DogezaUpEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.kiza_joint_pos = self._tensor(
            [KIZA_JOINTS[j] for j in self.robot.joint_names]
        )
        # 関節ごとの行動の倍率（腕だけ arm_action_scale、ほかは action_scale）
        arm = torch.tensor(
            [
                any(w in n for w in ("shoulder", "elbow", "wrist"))
                for n in self.robot.joint_names
            ],
            device=self.device,
        )
        self.scale = (
            cfg.action_scale + (cfg.arm_action_scale - cfg.action_scale) * arm.float()
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        """行動の基準を、ramp_s 秒かけて土下座から跪座へなめらかに移す（s = 0 → 1）。"""
        self.actions = actions.clone()
        s = torch.clamp(
            self.episode_length_buf.float() * self.step_dt / self.cfg.ramp_s, max=1.0
        )[:, None]
        s = s * s * (3 - 2 * s)  # 始めと終わりがゆっくりになる曲線
        base = self.goal_joint_pos + s * (self.kiza_joint_pos - self.goal_joint_pos)
        self.targets = base + self.scale * self.actions

    def _get_rewards(self) -> torch.Tensor:
        c, data, m = self.cfg, self.robot.data, self.m  # m は _get_dones で測ったもの
        g = data.projected_gravity_b.torch
        pitch = torch.atan2(g[:, 0], -g[:, 2])  # 腰の前後の傾き（+ 前、− 後ろ）[rad]
        lo, hi = c.pelvis_band
        band_err = torch.relu(lo - m["pelvis_z"]) + torch.relu(m["pelvis_z"] - hi)
        complete = (
            (m["head_z"] > c.done_head_min)
            & (torch.abs(pitch) < math.radians(c.done_pitch_deg))
            & (m["knee_z"].max(dim=-1).values < c.done_knee)
            & (m["heel_butt"] < c.done_heel_butt)
            & (m["pelvis_z"] > lo)
            & (m["pelvis_z"] < hi)
        ).float()
        terms = {
            "head_up": c.rew_head_up
            * torch.exp(-torch.relu(c.head_target - m["head_z"]) / c.head_sigma),
            "upright": c.rew_upright
            * torch.exp(-torch.square(pitch / math.radians(c.upright_sigma_deg))),
            "knees": c.rew_knees
            * torch.exp(-m["knee_z"] / c.height_sigma).mean(dim=-1),
            "heel_butt": c.rew_heel_butt
            * torch.exp(
                -torch.relu(m["heel_butt"] - c.heel_butt_full) / c.heel_butt_sigma
            ),
            "pelvis_height": c.rew_pelvis_height
            * torch.exp(-band_err / c.height_sigma),
            "complete": c.rew_complete * complete,
            "still": c.rew_still
            * complete
            * torch.exp(-0.05 * torch.sum(torch.square(data.joint_vel.torch), dim=-1)),
            "smooth": c.rew_smooth
            * torch.exp(
                -0.05
                * torch.sum(torch.square(self.actions - self.previous_actions), dim=-1)
            ),
        }
        self.reward_terms = {k: float(v.mean()) for k, v in terms.items()}
        self.complete_rate = float(complete.mean())
        return sum(terms.values())
