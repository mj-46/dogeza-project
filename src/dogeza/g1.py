from pathlib import Path

import omni.kit.app
from pxr import Gf, UsdGeom

# ---- ファイルの場所 ----
REPO_ROOT = Path(__file__).resolve().parents[2]
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
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*position))
    return prim
