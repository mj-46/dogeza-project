import time

from omni.kit.viewport.utility import get_active_viewport

from dogeza import g1, scene


def run(simulation_app, duration):
    g1_usd_path = g1.convert_urdf_to_usd()
    stage = scene.build_scene()
    g1.add_g1(stage, g1_usd_path)

    viewport = get_active_viewport()
    viewport.camera_path = "/World/Cameras/FrontLeft"

    print(f"[dogeza] {duration} 秒後に終了します。")
    start = time.time()
    while simulation_app.is_running() and time.time() - start < duration:
        simulation_app.update()
    print("[dogeza] 終了します。")
