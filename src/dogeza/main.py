import time

import omni.timeline
from omni.kit.viewport.utility import get_active_viewport

from dogeza.g1 import control as g1_control
from dogeza.g1 import keyframes, loader
from dogeza.scene import builder


def run(simulation_app, duration):
    g1_usd_path = loader.convert_urdf_to_usd()
    stage = builder.build_scene()
    loader.add_g1(stage, g1_usd_path)

    viewport = get_active_viewport()
    viewport.camera_path = "/World/Cameras/FrontLeft"

    # 物理の再生を開始
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    articulation = g1_control.get_articulation(stage)

    # 立つ → 土下座 → 立つ の動きを読み込む
    motion = keyframes.load_motion("unitree_g1_dogeza_motion.json")

    # 経過時間に応じた姿勢を毎フレーム書き込む（動きが終わったら最後の姿勢のまま）
    start = time.time()
    while simulation_app.is_running() and time.time() - start < duration:
        pose = keyframes.pose_at(motion, time.time() - start)
        g1_control.apply_pose(articulation, pose)
        simulation_app.update()
    print("[dogeza] 終了します。")
