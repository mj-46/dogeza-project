"""isaaclab_学習用"""

"""前半（立つ → 土下座）の学習環境の設定（数値・条件だけを書く。計算は env.py）。後半の設定はこれを引き継ぐ。"""

import json
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

# dogeza-project で URDF から変換した G1（先に scripts/run.py を 1 回実行して作っておく）
G1_USD = str(
    Path(__file__).resolve().parents[3]
    / "assets"
    / "robots"
    / "g1"
    / "g1_29dof_rev_1_0"
    / "g1_29dof_rev_1_0.usda"
)

# 保存した姿勢（assets/poses/）を使う。お手本の動きの JSON は使わない
POSES_DIR = Path(__file__).resolve().parents[3] / "assets" / "poses"


def load_pose(name):
    """assets/poses/{name}.json を読む（name = "stand" / "dogeza" / "kiza"）。"""
    with open(POSES_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


START = load_pose("stand")  # 立ち姿勢：初期姿勢と、観測の関節角度の基準
GOAL_JOINTS = load_pose("dogeza")[
    "joints"
]  # 土下座の最終姿勢：行動の基準（行動 0 = この姿勢を目標にする）


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
        # 関節のモーターの強さ（dogeza-project の physics_player.GAINS と同じ値にそろえる）
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
    """前半：立つ → 土下座。"""

    # ---- 時間 ----
    decimation = 4  # 物理 4 回に 1 回、方策が判断する（200Hz ÷ 4 = 50Hz）
    episode_length_s = (
        12.0  # 1 回の練習の長さ（土下座になるまで 5〜7 秒 + 保つ練習 5 秒以上）
    )
    time_obs_max_s = 12.0  # 観測の「経過時間」はこの秒数で 1 になり、その後は 1 のまま（再生を長くしても大丈夫なように）

    # ---- 観測と行動の数 ----
    # 観測 = 経過時間 1 + 重力の向き 3 + 腰の角速度 3 + 腰の高さ 1 + 関節角度 29 + 関節角速度 29 + 前回の行動 29
    observation_space = 95
    action_space = 29  # 29 関節の目標角度（土下座の最終姿勢 GOAL_JOINTS からの差）
    state_space = 0

    # ---- 物理（dogeza-project と同じ Isaac Sim PhysX・200Hz） ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )
    scene: DogezaSceneCfg = DogezaSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )
    # 行動 1.0 = 土下座の最終姿勢から 1 rad。行動 0 なら「土下座の姿勢になれ」という目標になる
    action_scale = 1.0

    # ---- 開始姿勢 ----
    start_stand_ratio = (
        0.6  # 立ち姿勢から始める割合（残りは土下座の最終姿勢から。完成を保つ練習）
    )
    start_joint_noise = 0.05  # 土下座から始めるとき、関節角度に加えるばらつき [rad]（立ち姿勢はばらつきなし＝本番と同じ）

    # ---- 報酬の重み（すべて 0 以上。倒れて練習が終わること自体を損にする。満点の合計 2.5） ----
    rew_heel_butt = 0.35  # かかと〜お尻の距離が近い
    rew_knees = 0.25  # 両膝が床に近い
    rew_pelvis_height = 0.15  # 腰の高さが土下座の範囲
    rew_palms = 0.1  # 両手のひらが床に近い
    rew_head = 0.0  # 頭が床に近い（ただし勢いよく下がっていると減らす）
    rew_complete = 1.0  # 土下座の条件がすべてそろった
    rew_still = 0.5  # 完成中に、関節が止まっている
    rew_lean = 0.05  # 前に倒れるきっかけ（腰が前に傾く・頭が下がる）
    rew_smooth = 0.1  # 行動がなめらか（震え防止）

    # 報酬の厳しさ
    heel_butt_full = 0.12  # かかと〜お尻がこれ以下 [m] で満点
    heel_butt_sigma = 0.1  # そこから 10 cm 離れると約 0.37
    height_sigma = 0.1  # 膝・手のひら・頭：高さ 5 cm で約 0.61、10 cm で約 0.37
    # 膝・手のひら・頭の報酬は「かかと〜お尻の報酬の割合（0〜1）」を掛ける（脚をたたまずに倒れても得をしないように）
    pelvis_band = (0.2, 0.35)  # 腰の高さ [m]（お手本の土下座は 0.26）
    head_speed_sigma = 0.5  # 頭が 0.5 m/s で下がっていると、頭の報酬が約 0.37 倍
    stand_head_height = 1.3  # 立ち姿勢の頭の高さ [m]（前に倒れるきっかけの計算用）

    # 「完成」の条件（すべて満たしたら完成）
    done_heel_butt = 0.15  # かかと〜お尻 [m] 以下
    done_knee = 0.03  # 両膝の高さ [m] 以下
    done_palm = 0.03  # 両手のひらの高さ [m] 以下
    done_head = 0.05  # 頭の高さ [m] 以下（腰の高さは pelvis_band の中）

    # ---- 終了条件（「倒れた」とみなす） ----
    max_back_tilt_deg = 10.0  # 腰が後ろに傾いた
    max_roll_deg = 30.0  # 横に傾いた
    max_yaw_deg = 45.0  # 体の正面の向きが変わった
    # 胴・腰・太ももが床についた（土下座へ倒れ込む途中で、腰は 0.15 m・胴は 0.20 m・太ももは 0.18 m まで下がるので、それより下）
    min_pelvis_z = 0.12
    min_torso_z = 0.15
    min_thigh_z = 0.12
    # 脚を伸ばしたまま顔から倒れた：腰が前に 70° 以上傾いているのに、かかと〜お尻が 0.3 m 以上離れている
    # （お手本では 70° 以上傾くのは土下座のときだけで、そのときかかと〜お尻は 0.14 m）
    face_down_pitch_deg = 70.0
    face_down_heel_butt = 0.3

    # ---- 再生（play.py）のときだけ使う設定 ----
    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = 4  # 見やすいように少なくする
        self.start_stand_ratio = 1.0  # 全員、立ち姿勢から
        self.episode_length_s = 30.0  # 動画用に長めに（土下座を保ち続ける）
