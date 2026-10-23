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

```bash
# 1. 取得
mkdir -p ~/workspace && cd ~/workspace
git clone https://github.com/j-matsui46/robot-dogeza.git
cd robot-dogeza

# 2. 実行（Isaac Sim が起動し、確認用シーンを表示して 5 秒後に終了）
~/isaacsim/python.sh scripts/run.py
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
- [ ] Unitree G1 を読み込んで表示する
- [ ] 土下座の姿勢データ（キーフレーム）を再生する
- [ ] 土下座の様子を動画で撮影し、YouTube に投稿する
- [ ] 強化学習で土下座に特化した全身制御を作る
- [ ] 自然言語の指示で土下座させる