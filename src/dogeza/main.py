import time

from dogeza import scene


def run(simulation_app, duration):

    scene.build_scene()

    print(f"[dogeza] {duration} 秒後に終了します。")
    start = time.time()
    while simulation_app.is_running() and time.time() - start < duration:
        simulation_app.update()
    print("[dogeza] 終了します。")
