"""前半（立つ → 土下座）の学習環境の本体。お手本はなぞらせず、土下座の姿勢そのものに報酬を与える。後半の環境はこれを引き継ぐ。"""

import math

import torch
from isaaclab.envs import DirectRLEnv
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from dogeza.g1.loader import CONTACT_BOXES
from dogeza_rl.first_half.env_cfg import GOAL_JOINTS, DogezaEnvCfg, load_pose

# 体の部位に固定した「測る点」（その部位から見た位置 [m]。URDF のメッシュから求めた）
HEEL = (
    -0.066,
    0.0,
    -0.035,
)  # かかと（足首の先の部位 ankle_roll_link。足の箱の後ろ下の端）
BUTT = (0.048, -0.043, -0.104)  # お尻（腰 pelvis のメッシュで、土下座のとき一番低い点）
HEAD = (
    0.076,
    0.005,
    0.456,
)  # 頭（胴 torso_link から見た、土下座のとき一番低い点＝おでこ）
KNEE = (
    (0.064, 0.007, 0.016),
    (0.064, -0.007, 0.016),
)  # 左右の膝の先端（すね knee_link）
PALM = (
    (0.066, -0.048, 0.010),
    (0.066, 0.048, 0.010),
)  # 左右の手のひらの中心（手 rubber_hand の箱の面）


class DogezaEnv(DirectRLEnv):
    """前半：立つ → 土下座。"""

    cfg: DogezaEnvCfg

    def __init__(self, cfg: DogezaEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]

        # 開始姿勢を、Isaac Lab の関節の並び順の tensor にする
        names = self.robot.joint_names

        def pose(name):
            p = load_pose(name)
            s, co = math.sin(p["pelvis_pitch"] / 2), math.cos(p["pelvis_pitch"] / 2)
            root = self._tensor(
                list(p["pelvis_position"]) + [0.0, s, 0.0, co]
            )  # 位置 + 向き (x, y, z, w)
            return root, self._tensor([p["joints"][j] for j in names])

        self.stand_root, self.stand_joints = pose("stand")
        self.dogeza_root, self.dogeza_joints = pose("dogeza")

        # 測る部位（左右の順）
        def body(*names_):
            return self.robot.find_bodies(list(names_), preserve_order=True)[0]

        self.feet_ids = body("left_ankle_roll_link", "right_ankle_roll_link")
        self.knee_ids = body("left_knee_link", "right_knee_link")
        self.hand_ids = body("left_rubber_hand", "right_rubber_hand")
        self.thigh_ids = body("left_hip_yaw_link", "right_hip_yaw_link")
        self.pelvis_id = body("pelvis")
        self.torso_id = body("torso_link")
        self.heel_local = self._tensor([HEEL, HEEL])
        self.butt_local = self._tensor([BUTT])
        self.head_local = self._tensor([HEAD])
        self.knee_local = self._tensor(KNEE)
        self.palm_local = self._tensor(PALM)

        self.default_joint_pos = (
            self.robot.data.default_joint_pos.torch.clone()
        )  # 立ち姿勢
        self.actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.goal_joint_pos = self._tensor([GOAL_JOINTS[j] for j in names]).repeat(
            self.num_envs, 1
        )  # 土下座の最終姿勢（行動の基準）
        self.prev_head_z = torch.zeros(self.num_envs, device=self.device)
        self.reward_terms, self.fall_causes = {}, {}  # TensorBoard 用の内訳
        self.complete_rate = 0.0

    def _tensor(self, data):
        return torch.tensor(data, dtype=torch.float, device=self.device)

    def _points(self, ids, local):
        """部位 ids に固定した点 local の位置（各 G1 の原点から見た位置）。(G1, 点, xyz)"""
        pos = self.robot.data.body_link_pos_w.torch[:, ids]
        q = self.robot.data.body_link_quat_w.torch[:, ids]  # (x, y, z, w)
        u, w = q[..., :3], q[..., 3:]
        t = 2.0 * torch.linalg.cross(u, local.expand_as(u))
        return (
            pos
            + local
            + w * t
            + torch.linalg.cross(u, t)
            - self.scene.env_origins[:, None]
        )

    # ---- 舞台 ----

    def _setup_scene(self):
        """手と足に箱形の当たり判定を足す（物理の開始前に呼ばれる）。"""
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
        self.targets = self.goal_joint_pos + self.cfg.action_scale * self.actions

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.targets)

    # ---- 観測（推論側でも、まったく同じ順番・計算で作ること） ----

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()
        data = self.robot.data
        elapsed = torch.clamp(
            self.episode_length_buf * self.step_dt / self.cfg.time_obs_max_s, max=1.0
        )  # 0 〜 1
        height = data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        obs = torch.cat(
            [
                elapsed.unsqueeze(-1),  # 1: 練習開始からの経過時間
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

    def _measure(self):
        """土下座の判定に使う量をまとめて測る。"""
        c = self.cfg
        butt = self._points(self.pelvis_id, self.butt_local)  # (G1, 1, xyz)
        heels = self._points(self.feet_ids, self.heel_local)  # (G1, 左右, xyz)
        m = {
            "heel_butt": torch.norm(heels - butt, dim=-1).mean(
                dim=-1
            ),  # 左右の平均の距離
            "knee_z": self._points(self.knee_ids, self.knee_local)[..., 2].clamp(
                min=0.0
            ),  # (G1, 左右)
            "palm_z": self._points(self.hand_ids, self.palm_local)[..., 2].clamp(
                min=0.0
            ),
            "head_z": self._points(self.torso_id, self.head_local)[:, 0, 2].clamp(
                min=0.0
            ),
            "pelvis_z": self.robot.data.root_link_pos_w.torch[:, 2]
            - self.scene.env_origins[:, 2],
        }
        lo, hi = c.pelvis_band
        m["complete"] = (
            (m["heel_butt"] < c.done_heel_butt)
            & (m["knee_z"].max(dim=-1).values < c.done_knee)
            & (m["palm_z"].max(dim=-1).values < c.done_palm)
            & (m["head_z"] < c.done_head)
            & (m["pelvis_z"] > lo)
            & (m["pelvis_z"] < hi)
        ).float()
        return m

    def _get_rewards(self) -> torch.Tensor:
        c, data, m = (
            self.cfg,
            self.robot.data,
            self.m,
        )  # m は _get_dones で測ったもの（終了判定の後に報酬が計算される）
        lo, hi = c.pelvis_band
        band_err = torch.relu(lo - m["pelvis_z"]) + torch.relu(
            m["pelvis_z"] - hi
        )  # 範囲の外にはみ出た長さ
        head_down_speed = (
            torch.relu(self.prev_head_z - m["head_z"]) / self.step_dt
        )  # 頭が下がる速さ [m/s]
        self.prev_head_z = m["head_z"]
        forward = data.projected_gravity_b.torch[:, 0].clamp(
            0.0, 1.0
        )  # 腰の前傾（0 = 直立、1 = 真下を向く）

        fold = torch.exp(
            -torch.relu(m["heel_butt"] - c.heel_butt_full) / c.heel_butt_sigma
        )  # 脚のたたみ具合（0〜1）

        def low(z):  # 床に近いほど 1（脚をたたんでいるときだけ）
            return torch.exp(-z / c.height_sigma) * (
                fold[:, None] if z.dim() == 2 else fold
            )

        terms = {
            "heel_butt": c.rew_heel_butt * fold,
            "knees": c.rew_knees * low(m["knee_z"]).mean(dim=-1),
            "pelvis_height": c.rew_pelvis_height
            * torch.exp(-band_err / c.height_sigma),
            "palms": c.rew_palms * low(m["palm_z"]).mean(dim=-1),
            "head": c.rew_head
            * low(m["head_z"])
            * torch.exp(-head_down_speed / c.head_speed_sigma),
            "complete": c.rew_complete * m["complete"],
            "still": c.rew_still
            * m["complete"]
            * torch.exp(-0.05 * torch.sum(torch.square(data.joint_vel.torch), dim=-1)),
            "lean": c.rew_lean
            * (
                0.5 * forward
                + 0.5 * (1.0 - m["head_z"] / c.stand_head_height).clamp(0.0, 1.0)
            ),
            "smooth": c.rew_smooth
            * torch.exp(
                -0.05
                * torch.sum(torch.square(self.actions - self.previous_actions), dim=-1)
            ),
        }
        self.reward_terms = {k: float(v.mean()) for k, v in terms.items()}
        self.complete_rate = float(m["complete"].mean())
        return sum(terms.values())

    # ---- 終了 ----

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        c, data = self.cfg, self.robot.data
        g = (
            data.projected_gravity_b.torch
        )  # 腰から見た重力の向き（前に傾くと x が +、横に傾くと y が ±）
        q = data.root_link_quat_w.torch  # (x, y, z, w)
        left_x = 2.0 * (
            q[:, 0] * q[:, 1] - q[:, 3] * q[:, 2]
        )  # 腰の「左向き」の矢印の x 成分（前後に傾いても変わらない）
        z = (
            data.body_link_pos_w.torch[:, :, 2] - self.scene.env_origins[:, 2:3]
        )  # 各部位の高さ
        self.m = self._measure()
        self.fall_causes = {
            "back_tilt": g[:, 0] < -math.sin(math.radians(c.max_back_tilt_deg)),
            "roll": torch.abs(g[:, 1]) > math.sin(math.radians(c.max_roll_deg)),
            "yaw": torch.abs(left_x) > math.sin(math.radians(c.max_yaw_deg)),
            "body_on_floor": (z[:, self.pelvis_id[0]] < c.min_pelvis_z)
            | (z[:, self.torso_id[0]] < c.min_torso_z)
            | (z[:, self.thigh_ids].min(dim=-1).values < c.min_thigh_z),
            "face_down": (g[:, 0] > math.sin(math.radians(c.face_down_pitch_deg)))
            & (self.m["heel_butt"] > c.face_down_heel_butt),
        }
        died = torch.stack(list(self.fall_causes.values())).any(dim=0)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    # ---- リセット ----

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 記録（TensorBoard）
        died = self.reset_terminated[env_ids]
        log = {
            "Episode/fall_rate": float(died.float().mean()),
            "Episode/complete_rate": self.complete_rate,
        }
        for (
            k,
            v,
        ) in (
            self.fall_causes.items()
        ):  # 倒れた G1 のうち、それぞれの条件に引っかかった割合
            log[f"FallCause/{k}"] = (
                float(v[env_ids][died].float().mean()) if died.any() else 0.0
            )
        log.update({f"Reward/{k}": v for k, v in self.reward_terms.items()})
        self.extras["log"] = log

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 開始姿勢：立ち姿勢（そのまま）か、土下座（関節に少しばらつき）。速さは 0
        k = len(env_ids)
        dogeza = (
            torch.rand(k, 1, device=self.device) >= self.cfg.start_stand_ratio
        ).float()  # 1 = 土下座から
        root = dogeza * self.dogeza_root + (1 - dogeza) * self.stand_root
        root[:, :3] += self.scene.env_origins[env_ids]
        noise = self.cfg.start_joint_noise * (
            2 * torch.rand(k, len(self.stand_joints), device=self.device) - 1
        )
        joints = (
            dogeza * (self.dogeza_joints + noise) + (1 - dogeza) * self.stand_joints
        )

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(k, 6, device=self.device), env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(position=joints, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(joints), env_ids=env_ids
        )
        self.prev_head_z[env_ids] = (
            0.0  # 最初の 1 回は「頭が下がる速さ」が 0 になるように（0 からは下がれない）
        )
