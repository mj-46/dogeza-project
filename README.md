# dogeza-project
自然言語で「土下座して」と指示すると、人型ロボットUnitree G1がIsaac Sim上で土下座するプロジェクト。強化学習で土下座に特化した全身制御を作成する。

## 動作環境

| 項目 | バージョン・内容 |
|---|---|
| OS | Ubuntu 24.04　LTS |
| Isaac Sim | 6.1.0（`~/isaacsim` にインストール） |
| Python | Isaac Sim 付属の Python 3.12（`~/isaacsim/python.sh`） |
| GPU | NVIDIA RTX 5070ti |

## 使い方

### 0. 事前準備：Isaac Sim のインストール

[NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) 6.1.0 を `~/isaacsim` にインストールしてください。
Python は Isaac Sim に付属のもの（`~/isaacsim/python.sh`）を使うため、他のインストールは不要です。

### 1. フォルダ配置

`robot-dogeza` と `unitree_ros` は**同じフォルダの中**に置いてください。

```
~/
├── isaacsim/            # Isaac Sim 本体
└── workspace/
    ├── dogeza-project/    # このリポジトリ
    └── unitree_ros/     # Unitree G1 のモデル（URDF）
```

### 2. 取得

```bash
mkdir -p ~/workspace && cd ~/workspace

# このリポジトリ
git clone https://github.com/j-matsui46/dogeza-project.git

# Unitree G1 のモデル（URDF）
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros.git
```

### 3. 実行

```bash
cd ~/workspace/dogeza-project
~/isaacsim/python.sh scripts/run.py --duration 10
```

## フォルダ構成

```
robot-dogeza/
├── assets/            # データ（土下座の姿勢など）
│   └── keyframes/
├── scripts/
│   └── run.py         # Isaac Sim の起動と終了だけを担当する入口
├── src/
│   └── dogeza/        # Isaac Sim 起動後の処理（シーン作成・ロボット・動作）
│       ├── __init__.py
│       ├── main.py    # 起動後に呼ばれる処理の本体
│       └── scene.py   # ステージ・床・照明・確認用オブジェクト・カメラ
├── pyproject.toml
├── LICENSE
└── README.md
```

## 現在の状況(仮)

- [x] Python スクリプトから Isaac Sim を起動・終了する
- [x] ステージを作成し、確認用のシーンを表示する
- [x] Unitree G1 を読み込んで表示する
- [ ] 土下座の姿勢データ（キーフレーム）を再生する
- [ ] 土下座の様子を動画で撮影し、YouTube に投稿する
- [ ] 強化学習で土下座に特化した全身制御を作る
- [ ] 自然言語の指示で土下座させる