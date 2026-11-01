# dogeza-project
人型ロボット Unitree G1 が Isaac Sim 上で土下座するプロジェクト。強化学習（PPO）で土下座に特化した全身制御を作成し、物理シミュレータ上で表示する。

- **前半**（立位 → 土下座）と**後半**（土下座 → 跪座）の 2 つの方策（.pt）を別々に学習する
- **姿勢そのもの**（膝・手のひら・頭が床につく、かかととお尻が近い など）に報酬を与える
- 推論は .pt と経過時間だけで動く

## 動作環境

| 項目 | バージョン・内容 |
|---|---|
| OS | Ubuntu 24.04 LTS |
| Isaac Sim | 6.2.0（`~/isaacsim` にインストール） |
| Isaac Lab | 3.0（学習時に使用。`~/IsaacLab` にインストール） |
| 強化学習 | RSL-RL（PPO。Isaac Lab に付属） |
| Python | Isaac Sim 付属の Python 3.12（`~/isaacsim/python.sh`） |
| GPU | NVIDIA RTX 5070 Ti |
| その他 | ffmpeg（録画に使用。`sudo apt install ffmpeg`） |

## 使い方

### 0. 事前準備

1. [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) 6.2.0 を `~/isaacsim` にインストールする
2. [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 3.0 を `~/IsaacLab` にインストールする（学習するときだけ必要）
3. 録画に使う ffmpeg をインストールする：`sudo apt install ffmpeg`

Python は Isaac Sim に付属のものを使うため、ほかのインストールは不要です。

### 1. フォルダ配置

`dogeza-project` と `unitree_ros` は**同じフォルダの中**に置いてください。

```
~/
├── isaacsim/              # Isaac Sim 本体
├── IsaacLab/              # Isaac Lab（学習するときだけ）
└── workspace/
    ├── dogeza-project/    # このリポジトリ
    └── unitree_ros/       # Unitree G1 のモデル（URDF）
```

### 2. 取得

```bash
mkdir -p ~/workspace && cd ~/workspace

# このリポジトリ
git clone https://github.com/mj-46/dogeza-project.git

# Unitree G1 のモデル（URDF）
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros.git
```

### 3. 実行（学習済みの方策で土下座させる）

`assets/policies/` の 2 つの方策で、「立位で静止 3 秒 → 土下座 10 秒 → 跪座 10 秒」（1 テイク 23 秒）を、カメラを変えながら 3 テイク行います。

```bash
cd ~/workspace/dogeza-project
~/isaacsim/python.sh scripts/run.py --duration 70            # 画面で見る
~/isaacsim/python.sh scripts/run.py --duration 70 --record   # MP4 に録画する（60 fps）
```

- 初回は URDF から G1 の USD を自動で作ります（`assets/robots/g1/`）

### 4. 学習（Isaac Lab）

前半・後半のどちらを学習するかを `--task` で指定します。

```bash
cd ~/workspace/dogeza-project
~/IsaacLab/_isaac_sim/python.sh scripts/train.py --task dogeza-first-half --num_envs 4096    # 前半：立位 → 土下座
~/IsaacLab/_isaac_sim/python.sh scripts/train.py --task dogeza-second-half --num_envs 4096   # 後半：土下座 → 跪座
```

学習結果は `logs/rsl_rl/dogeza_g1_down/`（前半）・`logs/rsl_rl/dogeza_g1_up/`（後半）に、200 回ごとに保存されます。

## フォルダ構成

```
dogeza-project/
├── assets/
│   ├── poses/                     # 姿勢（学習・推論で使う）
│   │   ├── stand.json             #   立位（開始姿勢・観測の基準）
│   │   ├── dogeza.json            #   土下座の最終姿勢（前半の行動の基準）
│   │   └── kiza.json              #   跪座（後半の行き先）
│   ├── policies/                  # 学習済みの方策
│   │   ├── dogeza_first_half.pt   #   前半：立位 → 土下座
│   │   └── dogeza_second_half.pt  #   後半：土下座 → 跪座
│   └── robots/g1/                 # 変換した G1 の USD（自動生成・Git 管理外）
├── logs/                          # 学習結果（Git 管理外）
├── scripts/
│   ├── run.py                     # Isaac Sim の起動と終了だけを担当する入口
│   ├── train.py                   # 学習（--task で前半・後半を指定）
│   └── play.py                    # 学習結果の再生と方策の書き出し（--task で前半・後半を指定）
└── src/
    ├── dogeza/                    # 土下座プロジェクト本体（Isaac Sim 起動後の処理）
    │   ├── __init__.py 
    │   ├── main.py                # 全体の流れ（舞台 → G1 → 動作 → 実行・録画）
    │   ├── scene/
        │   ├── __init__.py 
    │   │   ├── builder.py         # ステージ・物理・床・照明・カメラ
    │   │   └── runner.py          # 実際の経過時間で動かす（録画しないとき）
    │   └── g1/
        │   ├── __init__.py 
    │       ├── loader.py          # URDF → USD 変換、G1 と手足の当たり判定の追加
    │       ├── control.py         # 関節の操作（Articulation の取得、姿勢の設定）
    │       ├── keyframes.py       # キーフレームの読み込みと補間
    │       ├── physics_player.py  # 姿勢を目標に、物理で動かす
    │       └── policy_player.py   # 学習した方策で動かす（前半 → 後半、カメラを変えて 3 テイク）
    ├── dogeza_rl/                 # 強化学習（Isaac Lab のタスク）
    │   ├── __init__.py            # 2 つのタスクの登録
    │   ├── first_half/            # 前半：立位 → 土下座
    │   │   ├── env_cfg.py         #   設定（報酬の重み・終了条件など）
    │   │   ├── env.py             #   環境（観測・行動・報酬・終了・リセット）
    │   │   └── agent_cfg.py       #   PPO の設定
    │   └── second_half/           # 後半：土下座 → 跪座（前半を引き継ぎ、違うところだけを書く）
    │       ├── env_cfg.py
    │       ├── env.py
    │       └── agent_cfg.py
    └── isaac_video/               # 録画機能（土下座に依存しない独立した部品）
        └── recorder.py            # カメラ映像を MP4 に保存する
```