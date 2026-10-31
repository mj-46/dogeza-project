"""isaac sim_再生用"""

import time


def run_realtime(simulation_app, step, duration):
    """step(t) を呼びながら、実際の経過時間で duration 秒進める（処理が重いとコマが飛ぶ）。"""
    start = time.time()
    while simulation_app.is_running() and time.time() - start < duration:
        step(time.time() - start)
        simulation_app.update()
