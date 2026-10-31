"""isaac sim_再生用"""

from pathlib import Path

import omni.timeline
from omni.kit.viewport.utility import get_active_viewport

from dogeza.g1 import control, loader, policy_player
from dogeza.scene import builder, runner
from isaac_video.recorder import VideoRecorder

CAMERA = "/World/Cameras/Record"
VIDEO_DIR = Path(__file__).resolve().parents[2] / "videos"
ROOT_ASSIST = 0.1  # 腰の補助（0 = 純粋な物理, 1 = 強く補助）


def run(simulation_app, duration, record=False):
    g1_usd_path = loader.convert_urdf_to_usd()
    stage = builder.build_scene()
    loader.add_g1(stage, g1_usd_path)
    loader.add_contact_shapes(stage)

    viewport = get_active_viewport()
    if viewport is not None:
        viewport.camera_path = CAMERA

    # 物理の再生を開始
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    articulation = control.get_articulation(stage)

    # policyでの再生
    step = policy_player.PolicyPlayer(articulation)

    # 実行（録画するかどうかで、時間の進め方だけが変わる）
    if record:
        with VideoRecorder(CAMERA, VIDEO_DIR, name="dogeza") as recorder:
            recorder.run(simulation_app, step, duration)
    else:
        runner.run_realtime(simulation_app, step, duration)
