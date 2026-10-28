"""土下座の学習環境の本体（観測・行動・報酬・終了・リセット）。"""

import torch
from isaaclab.envs import DirectRLEnv
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from dogeza.g1 import keyframes
from dogeza.g1.loader import CONTACT_BOXES
from dogeza_rl.env_cfg import DogezaEnvCfg


class DogezaEnv(DirectRLEnv):
    cfg: DogezaEnvCfg

    def __init__(self, cfg: DogezaEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]

        # お手本を「方策の判断の間隔（step_dt = 0.02 秒）ごとの表」にして GPU に置く
        motion = keyframes.load_motion(cfg.motion_file)
        self.num_frames = int(round(float(motion["times"][-1]) / self.step_dt)) + 1
        poses = [
            keyframes.pose_at(motion, i * self.step_dt) for i in range(self.num_frames)
        ]
        names = self.robot.joint_names  # Isaac Lab 側の関節の並び順に合わせる
        self.ref_joint_pos = self._tensor(
            [[p["joints"][n] for n in names] for p in poses]
        )
        self.ref_root_pos = self._tensor([p["pelvis_position"] for p in poses])
        pitch = self._tensor([p["pelvis_pitch"] for p in poses])
        zero = torch.zeros_like(pitch)
        self.ref_root_quat = torch.stack(
            [zero, torch.sin(pitch / 2), zero, torch.cos(pitch / 2)], dim=-1
        )  # (x, y, z, w)
        self.ref_gravity = torch.stack(
            [torch.sin(pitch), zero, -torch.cos(pitch)], dim=-1
        )  # 腰から見た重力の向き

        self.default_joint_pos = (
            self.robot.data.default_joint_pos.torch.clone()
        )  # 立ち姿勢
        self.actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.start_frame = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _tensor(self, data):
        return torch.tensor(data, dtype=torch.float, device=self.device)

    def _frame(self):
        """各 G1 が今、お手本の何コマ目にいるか。"""
        return torch.clamp(
            self.start_frame + self.episode_length_buf, max=self.num_frames - 1
        )

    # ---- 舞台 ----

    def _setup_scene(self):
        """dogeza-project と同じく、手と足に箱形の当たり判定を足す（物理の開始前に呼ばれる）。"""
        stage = self.sim.stage
        for env_path in self.scene.env_prim_paths:
            robot = stage.GetPrimAtPath(f"{env_path}/Robot")
            if not robot.IsValid():
                continue
            links = {
                p.GetName(): p
                for p in Usd.PrimRange(robot)
                if p.HasAPI(UsdPhysics.RigidBodyAPI)
            }
            for name, (center, size) in CONTACT_BOXES.items():
                link = links[name]
                if link.IsInstance():
                    link.SetInstanceable(False)
                box = UsdGeom.Cube.Define(
                    stage, link.GetPath().AppendChild("ContactBox")
                )
                box.CreateSizeAttr(1.0)
                box.CreatePurposeAttr(UsdGeom.Tokens.guide)
                xf = UsdGeom.XformCommonAPI(box)
                xf.SetTranslate(Gf.Vec3d(*center))
                xf.SetScale(Gf.Vec3f(*size))
                UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    # ---- 行動 ----

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        self.targets = self.default_joint_pos + self.cfg.action_scale * self.actions

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.targets)

    # ---- 観測（推論側の PolicyPlayer でも、まったく同じ順番・計算で作ること） ----

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()
        data = self.robot.data
        phase = self._frame().float() / (self.num_frames - 1)  # 0（開始）〜 1（終了）
        height = data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        obs = torch.cat(
            [
                phase.unsqueeze(-1),  # 1: 時刻の位相
                data.projected_gravity_b.torch,  # 3: 腰から見た重力の向き（＝腰の傾き）
                data.root_link_ang_vel_b.torch,  # 3: 腰の角速度
                height.unsqueeze(-1),  # 1: 腰の高さ
                data.joint_pos.torch
                - self.default_joint_pos,  # 29: 関節角度（立ち姿勢からの差）
                data.joint_vel.torch,  # 29: 関節角速度
                self.actions,  # 29: 前回の行動
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ---- 報酬 ----

    def _get_rewards(self) -> torch.Tensor:
        data = self.robot.data
        frame = self._frame()
        height = data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]

        # お手本とのずれ（小さいほど良い）
        pose_error = torch.sum(
            torch.square(data.joint_pos.torch - self.ref_joint_pos[frame]), dim=-1
        )
        height_error = torch.square(height - self.ref_root_pos[frame, 2])
        orientation_error = torch.sum(
            torch.square(data.projected_gravity_b.torch - self.ref_gravity[frame]),
            dim=-1,
        )
        action_rate = torch.sum(
            torch.square(self.actions - self.previous_actions), dim=-1
        )

        # exp(-ずれ) は、ずれ 0 で 1、ずれが大きいほど 0 に近づく
        return (
            self.cfg.rew_pose * torch.exp(-pose_error)
            + self.cfg.rew_height * torch.exp(-50.0 * height_error)
            + self.cfg.rew_orientation * torch.exp(-10.0 * orientation_error)
            + self.cfg.rew_action_rate * action_rate
        )

    # ---- 終了 ----

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self._frame()
        height = (
            self.robot.data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        )
        died = (
            torch.abs(height - self.ref_root_pos[frame, 2]) > self.cfg.max_height_error
        )  # 倒れた
        time_out = (
            self.start_frame + self.episode_length_buf >= self.num_frames - 1
        ) | (
            self.episode_length_buf >= self.max_episode_length - 1
        )  # お手本の最後まで到達
        return died, time_out

    # ---- リセット ----

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # お手本のどの時刻から始めるか
        if self.cfg.random_start:
            self.start_frame[env_ids] = torch.randint(
                0, self.num_frames - 1, (len(env_ids),), device=self.device
            )
        else:
            self.start_frame[env_ids] = 0
        frame = self.start_frame[env_ids]
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        # その時刻のお手本の姿勢に置き、動きを止める
        root_pose = torch.cat(
            [
                self.ref_root_pos[frame] + self.scene.env_origins[env_ids],
                self.ref_root_quat[frame],
            ],
            dim=-1,
        )
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(env_ids), 6, device=self.device),
            env_ids=env_ids,
        )
        self.robot.write_joint_position_to_sim_index(
            position=self.ref_joint_pos[frame], env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self.ref_joint_pos[frame]), env_ids=env_ids
        )
