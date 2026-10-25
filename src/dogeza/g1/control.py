import math

import numpy as np
from pxr import Usd, UsdPhysics


def get_articulation(stage):
    """G1の関節をまとめて操作するためのオブジェクトを作る"""
    from isaacsim.core.experimental.prims import Articulation

    # /World/G1の中から、関節まとまりの根元（ArticulationRootAPIを持つプリム）を探す
    g1 = stage.GetPrimAtPath("/World/G1")
    roots = [
        p
        for p in Usd.PrimRange(g1, Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not roots:
        raise RuntimeError(
            "/World/G1 の中に関節の根元（ArticulationRootAPI）が見つかりません。"
        )
    root = roots[0]
    articulation = Articulation(str(root.GetPath()))
    print(f"[dogeza] 関節の根元: {root.GetPath()}（関節数 {articulation.num_dofs}）")
    return articulation


def apply_pose(articulation, pose):
    """姿勢データを G1 に書き込む（腰の位置・傾きと、全関節の角度）。"""
    # 腰の位置と傾き（Y軸まわりの回転。クォータニオンは w, x, y, z の順）
    pitch = pose["pelvis_pitch"]
    position = np.array([pose["pelvis_position"]])
    orientation = np.array([[math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]])
    articulation.set_world_poses(positions=position, orientations=orientation)

    # 関節の角度（G1側の関節の並び順に合わせる。データにない関節は0）
    angles = np.array(
        [[pose["joints"].get(name, 0.0) for name in articulation.dof_names]]
    )
    articulation.set_dof_positions(angles)
    # 関節のモーターが元に戻そうとしないように
    articulation.set_dof_position_targets(angles)

    # 動きを止める（速度を0に）
    articulation.set_dof_velocities(np.zeros_like(angles))
    articulation.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
