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
dogeza-project/
├── assets/
│ ├── keyframes/
│ │ └── unitree_g1_dogeza_motion.json # 土下座のお手本（0.1 秒ごとの姿勢）
│ └── robots/g1/ # 変換した G1 の USD（自動生成・Git 管理外）
├── scripts/
│ └── run.py # Isaac Sim の起動と終了だけを担当する入口
└── src/
├── dogeza/ # 土下座プロジェクト本体（Isaac Sim 起動後の処理）
│ ├── main.py # 全体の流れ（舞台 → G1 → 動作 → 実行）
│ ├── scene/
│ │ ├── builder.py # ステージ・物理・床・照明・カメラ
│ │ └── runner.py # 実際の経過時間で動かす（録画しないとき）
│ └── g1/
│ ├── loader.py # URDF → USD 変換、G1 と手足の当たり判定の追加
│ ├── control.py # 関節の操作（Articulation の取得、姿勢の設定）
│ ├── keyframes.py # キーフレームの読み込みと補間
│ ├── kinematics_player.py # 姿勢をそのまま再生する
│ └── physics_player.py # 姿勢を目標に、物理で動かす
└── isaac_video/ # 録画機能（土下座に依存しない独立した部品）
└── recorder.py # カメラ映像を MP4 に保存する
```

## 現在の状況(仮)

- [x] Python スクリプトから Isaac Sim を起動・終了する
- [x] ステージを作成し、確認用のシーンを表示する
- [x] Unitree G1 を読み込んで表示する
- [x] 土下座の姿勢データ（キーフレーム）をキネマティックに再生する
- [x] 土下座の様子を動画で撮影する
- [x] キネマティックな再生と同じ動きを物理で実際に動かす
- [ ] 強化学習で土下座に特化した全身制御を作る
- [ ] 自然言語の指示で土下座させる