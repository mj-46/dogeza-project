"""後半（土下座 → 跪座）の学習環境の設定。前半の設定（first_half/env_cfg.py）を引き継ぎ、違うところだけを書く。"""

from isaaclab.utils import configclass

from dogeza_rl.first_half.env_cfg import DogezaEnvCfg, load_pose

KIZA_JOINTS = load_pose("kiza")["joints"]  # 跪座：行動の基準の行き先（土下座 → 跪座）


@configclass
class DogezaUpEnvCfg(DogezaEnvCfg):
    """後半：土下座 → 跪座。前半と違うところだけを書く（ほかは前半と同じ）。"""

    # ---- 時間 ----
    episode_length_s = 10.0  # 起き上がるまで約 3 秒 + 跪座を保つ練習
    time_obs_max_s = 10.0

    # ---- 行動の基準：ramp_s 秒かけて、土下座から跪座へなめらかに移す（行動 0 ならこれに沿って起き上がる） ----
    ramp_s = 3.0
    # 行動 1.0 = 基準から何 rad 動かすか。脚・腰は前半と同じ 1 rad、腕だけ 0.1 rad に小さくする
    # （腕は基準＝「手のひらを太ももに置いた跪座の腕」からほとんど動かせない → 手は太ももの上、腕が後ろに回らない）
    action_scale = 1.0
    arm_action_scale = 0.1

    # ---- 開始姿勢：全員、土下座の最終姿勢から（関節に少しばらつき。前半の方策が終わった状態を想定） ----
    start_stand_ratio = 0.0

    # ---- 報酬の重み（すべて 0 以上。満点の合計 2.6） ----
    rew_head_up = 0.35  # 頭が跪座の高さまで上がった
    rew_upright = 0.25  # 腰がまっすぐ立っている（前後の傾きが小さい）
    rew_knees = 0.15  # 両膝が床についたまま
    rew_heel_butt = 0.15  # お尻がかかとの上にある（かかと〜お尻が近い）
    rew_pelvis_height = 0.1  # 腰の高さが跪座の範囲
    rew_complete = 1.0  # 跪座の条件がすべてそろった
    rew_still = 0.5  # 完成中に、関節が止まっている
    rew_smooth = 0.1  # 行動がなめらか（震え防止）
    rew_palms = 0.0  # （前半用。後半では使わない）
    rew_head = 0.0
    rew_lean = 0.0

    # 報酬の厳しさ（MuJoCo で跪座を保ったときの値：頭 0.84 m・腰 0.33 m・かかと〜お尻 0.16 m・腰の傾き −2°）
    head_target = 0.82  # 頭の高さ [m]。これ以上で満点
    head_sigma = 0.2  # そこから 20 cm 低いと約 0.37
    upright_sigma_deg = 10.0  # 腰が 10° 傾くと約 0.37
    heel_butt_full = 0.17  # かかと〜お尻がこれ以下 [m] で満点
    pelvis_band = (0.28, 0.40)  # 腰の高さ [m]

    # 「完成」の条件（すべて満たしたら完成）
    done_head_min = 0.75  # 頭の高さ [m] 以上
    done_pitch_deg = 12.0  # 腰の前後の傾きが ±12° 以内
    done_knee = 0.03  # 両膝の高さ [m] 以下
    done_heel_butt = 0.2  # かかと〜お尻 [m] 以下（腰の高さは pelvis_band の中）

    # ---- 終了条件：後ろへの傾きだけ前半よりゆるめる ----
    max_back_tilt_deg = 20.0

    def play_mode(self):
        super().play_mode()
        self.start_stand_ratio = 0.0  # 再生も土下座から
