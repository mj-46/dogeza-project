from dogeza.g1 import control, keyframes


class KeyframePlayer:
    """KeyframePlayer(articulation, motion)(t) で、時刻 t [秒] の姿勢を G1 に書き込む。"""

    def __init__(self, articulation, motion):
        self.articulation = articulation
        self.motion = motion

    def __call__(self, t):
        control.apply_pose(self.articulation, keyframes.pose_at(self.motion, t))
