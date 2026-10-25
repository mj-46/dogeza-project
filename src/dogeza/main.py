from pathlib import Path

import omni.timeline
from omni.kit.viewport.utility import get_active_viewport

from dogeza.g1 import control, keyframes, loader, player
from dogeza.scene import builder, runner
from isaac_video.recorder import VideoRecorder

CAMERA = "/World/Cameras/FrontLeft"
VIDEO_DIR = Path(__file__).resolve().parents[1] / "isaac_video" / "output"


def run(simulation_app, duration, record=False):
    g1_usd_path = loader.convert_urdf_to_usd()
    stage = builder.build_scene()
    loader.add_g1(stage, g1_usd_path)

    viewport = get_active_viewport()
    viewport.camera_path = "/World/Cameras/FrontLeft"

    # 物理の再生を開始
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    articulation = control.get_articulation(stage)

    # 立つ → 土下座 → 立つ の動きを読み込む
    motion = keyframes.load_motion("unitree_g1_dogeza_motion.json")
    step = player.KeyframePlayer(articulation, motion)

    # 実行（録画するかどうかで、時間の進め方だけが変わる）
    if record:
        with VideoRecorder(CAMERA, VIDEO_DIR, name="dogeza") as recorder:
            recorder.run(simulation_app, step, duration)
    else:
        runner.run_realtime(simulation_app, step, duration)
