from pathlib import Path

import omni.kit.app
from pxr import Gf, UsdGeom

# ---- ファイルの場所 ----
REPO_ROOT = Path(__file__).resolve().parents[3]
# Unitree 公式の G1 モデル（リポジトリと同じ階層の unitree_ros にクローンしておく）
G1_URDF = (
    REPO_ROOT.parent
    / "unitree_ros"
    / "robots"
    / "g1_description"
    / "g1_29dof_rev_1_0.urdf"
)
# URDF から変換した USD の置き場所（自動生成）
G1_USD_DIR = REPO_ROOT / "assets" / "robots" / "g1"


def convert_urdf_to_usd():
    """G1 の URDF を USD に変換する。変換済みならそれを使う（初回だけ時間がかかる）。"""
    existing = sorted(G1_USD_DIR.rglob(f"{G1_URDF.stem}*.usd*"))
    if existing:
        return str(existing[0])
    if not G1_URDF.exists():
        raise FileNotFoundError(f"G1 の URDF が見つかりません: {G1_URDF}")

    # URDF インポーター拡張を有効にしてから読み込む
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.asset.importer.urdf", True
    )
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    G1_USD_DIR.mkdir(parents=True, exist_ok=True)
    config = URDFImporterConfig(
        urdf_path=str(G1_URDF), usd_path=str(G1_USD_DIR), fix_base=False
    )
    # URDF を USD に変換
    usd_path = URDFImporter(config).import_urdf()
    return usd_path


def add_g1(stage, usd_path, position=(0.0, 0.0, 0.793)):
    """変換した G1 をステージに配置する（原点に立ち、+X 方向を向く）。"""
    prim = stage.DefinePrim("/World/G1", "Xform")
    prim.GetReferences().AddReference(usd_path)
    prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*position))
    return prim


# 当たり判定を足すリンクと、その箱の中心・大きさ（リンク座標系, m）
# 元の URDF では手に当たり判定がなく、足は足裏の小さな球 4 つだけなので、床に手・足の甲がつく土下座ではすり抜ける
CONTACT_BOXES = {
    "left_rubber_hand": ((0.066, -0.0145, 0.010), (0.132, 0.067, 0.106)),
    "right_rubber_hand": ((0.066, 0.0145, 0.010), (0.132, 0.067, 0.106)),
    "left_ankle_roll_link": ((0.038, 0.0, -0.0055), (0.208, 0.076, 0.059)),
    "right_ankle_roll_link": ((0.038, 0.0, -0.0055), (0.208, 0.076, 0.059)),
}


def add_contact_shapes(stage):
    """手と足に箱形の当たり判定を足す（物理の再生を始める前に呼ぶ）。"""
    from pxr import Usd, UsdPhysics

    g1 = stage.GetPrimAtPath("/World/G1")
    links = {
        p.GetName(): p for p in Usd.PrimRange(g1) if p.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    for name, (center, size) in CONTACT_BOXES.items():
        if name not in links:
            print(
                f"[dogeza] リンク {name} が見つからないため、当たり判定を追加できませんでした。"
            )
            continue
        if links[name].IsInstance():  # インスタンス（書き換え禁止）なら
            links[name].SetInstanceable(False)  # 共有をやめて、書き換えられるようにする
        box = UsdGeom.Cube.Define(
            stage, links[name].GetPath().AppendChild("ContactBox")
        )
        box.CreateSizeAttr(1.0)
        box.CreatePurposeAttr(UsdGeom.Tokens.guide)  # 見た目には表示しない
        xf = UsdGeom.XformCommonAPI(box)
        xf.SetTranslate(Gf.Vec3d(*center))
        xf.SetScale(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    print(
        f"[dogeza] 手と足に当たり判定を追加しました: {[n for n in CONTACT_BOXES if n in links]}"
    )
