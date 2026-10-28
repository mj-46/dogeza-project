"""土下座の学習環境の設定（数値・条件だけを書く。計算は env.py）。"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from dogeza.g1 import keyframes

# dogeza-project で URDF から変換した G1（先に scripts/run.py を 1 回実行して作っておく）
G1_USD = str(
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "g1"
    / "g1_29dof_rev_1_0"
    / "g1_29dof_rev_1_0.usda"
)
MOTION_FILE = "unitree_g1_dogeza_motion.json"

# お手本の 0 秒の姿勢（立ち姿勢）。初期姿勢と、行動の基準（行動 = この姿勢からの差）に使う
START = keyframes.pose_at(keyframes.load_motion(MOTION_FILE), 0.0)


@configclass
class DogezaSceneCfg(InteractiveSceneCfg):
    """1 体分の舞台。これが num_envs 体分コピーされる。"""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=G1_USD, variants={"Physics": "physx"}),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in START["pelvis_position"]),
            # Isaac Lab のクォータニオンは (x, y, z, w) の順（dogeza-project の (w, x, y, z) と逆なので注意）
            rot=(
                0.0,
                math.sin(START["pelvis_pitch"] / 2),
                0.0,
                math.cos(START["pelvis_pitch"] / 2),
            ),
            joint_pos={name: float(angle) for name, angle in START["joints"].items()},
        ),
        # 関節のモーターの強さ（physics_player.GAINS と同じ値にそろえる）
        actuators={
            "hip": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*"], stiffness=150.0, damping=5.0
            ),
            "knee": ImplicitActuatorCfg(
                joint_names_expr=[".*_knee_joint"], stiffness=200.0, damping=6.0
            ),
            "ankle": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_.*"], stiffness=40.0, damping=2.0
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"], stiffness=200.0, damping=5.0
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint"],
                stiffness=50.0,
                damping=2.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_.*"], stiffness=20.0, damping=1.0
            ),
        },
    )
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        collision_group=-1,  # すべての G1 と当たる
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            )
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0)
    )


@configclass
class DogezaEnvCfg(DirectRLEnvCfg):
    # ---- 時間 ----
    decimation = 4  # 物理 4 回に 1 回、方策が判断する（200Hz ÷ 4 = 50Hz）
    episode_length_s = 20.0  # 1 回の練習の最大の長さ

    # ---- 観測と行動の数 ----
    # 観測 = 位相 1 + 重力の向き 3 + 腰の角速度 3 + 腰の高さ 1 + 関節角度 29 + 関節角速度 29 + 前回の行動 29
    observation_space = 95
    action_space = 29  # 29 関節の目標角度（立ち姿勢からの差）
    state_space = 0

    # ---- 物理（dogeza-project と同じ Isaac Sim PhysX・200Hz） ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )

    # ---- 舞台（並列数と、隣の G1 との間隔 [m]） ----
    scene: DogezaSceneCfg = DogezaSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )

    # ---- お手本と行動 ----
    motion_file = MOTION_FILE
    action_scale = 1.0  # 行動 1.0 = 立ち姿勢から 1 rad
    random_start = True  # お手本のランダムな時刻の姿勢から練習を始める（途中の「土下座中」からも練習する）

    # ---- 報酬の重み ----
    rew_pose = 0.5  # 関節角度がお手本に近い
    rew_height = 0.2  # 腰の高さがお手本に近い
    rew_orientation = 0.2  # 腰の傾きがお手本に近い
    rew_action_rate = -0.01  # 行動の急な変化（ガタガタした動き）への罰

    # ---- 終了条件 ----
    max_height_error = 0.3  # 腰の高さがお手本から 0.3 m 以上ずれたら「倒れた」とみなす

    # ---- 再生（play.py）のときだけ使う設定 ----
    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = 4  # 見やすいように少なくする
        self.random_start = False  # お手本の最初（立ち姿勢）から通して動かす
