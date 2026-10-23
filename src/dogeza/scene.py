"""シーン（舞台）の作成：ステージ・物理・床・照明・撮影用カメラ。"""

import omni.usd
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

# ---- プリムのパス ----
WORLD = "/World"
PHYSICS_SCENE = "/World/PhysicsScene"
GROUND = "/World/Ground"
GROUND_MATERIAL = "/World/Physics/GroundMaterial"
LIGHTS = "/World/Lights"
CAMERAS = "/World/Cameras"


def new_stage():
    """空のステージを作り、Z軸を上・長さをメートル・質量をキログラムにする。"""
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, WORLD)
    return stage


def add_physics_scene(stage, steps_per_second=200):
    """重力と物理計算の刻み（1秒あたりの計算回数）を設定する。"""
    scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateTimeStepsPerSecondAttr(steps_per_second)


def add_ground(stage, size=20.0, thickness=0.02, friction=1.0):
    """当たり判定と摩擦のある床。上面が z=0 になるように置く。"""
    ground = UsdGeom.Cube.Define(stage, GROUND)
    ground.CreateSizeAttr(1.0)
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.45, 0.45)])
    xf = UsdGeom.XformCommonAPI(ground)
    xf.SetTranslate(Gf.Vec3d(0.0, 0.0, -thickness / 2))
    xf.SetScale(Gf.Vec3f(size, size, thickness))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())  # 当たり判定

    # 物理マテリアル（摩擦・反発）を作って床に割り当てる
    material = UsdShade.Material.Define(stage, GROUND_MATERIAL)
    phys = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys.CreateStaticFrictionAttr(friction)
    phys.CreateDynamicFrictionAttr(friction)
    phys.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def add_lights(stage):
    """全体を照らす環境光と、影をつける平行光。"""
    UsdGeom.Xform.Define(stage, LIGHTS)
    dome = UsdLux.DomeLight.Define(stage, f"{LIGHTS}/Dome")
    dome.CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, f"{LIGHTS}/Sun")
    sun.CreateIntensityAttr(3000.0)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-50.0, 0.0, 35.0))


def _add_camera(stage, path, eye, target, focal_length=24.0):
    """eye の位置から target を見るカメラを置く。"""
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal_length)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0)
    )
    cam.AddTransformOp().Set(view.GetInverse())
    return cam


def add_cameras(stage, target=(0.0, 0.0, 0.5)):
    """撮影用カメラ。ロボットは原点で +X 方向を向く想定。"""
    UsdGeom.Xform.Define(stage, CAMERAS)
    _add_camera(
        stage, f"{CAMERAS}/Side", eye=(0.0, -4.0, 0.8), target=target
    )  # 真横（左側から）
    _add_camera(
        stage, f"{CAMERAS}/FrontLeft", eye=(3.0, 2.0, 1.6), target=target
    )  # 斜め前


def build_scene():
    stage = new_stage()
    add_physics_scene(stage)
    add_ground(stage)
    add_lights(stage)
    add_cameras(stage)
    return stage
