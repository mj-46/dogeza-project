"""isaaclab_学習用"""

"""PPO（RSL-RL）の学習方法の設定。Isaac Lab 付属タスクの設定を参考にしている。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DogezaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # 1 回の更新の前に、1 体あたり何ステップ経験を集めるか（env_cfg.py の steps_per_iter と同じ値）
    max_iterations = 6000  # 更新の回数
    save_interval = 200  # 何回ごとに途中経過（.pt）を保存するか
    experiment_name = "dogeza_g1_down"  # 保存先: logs/rsl_rl/dogeza_g1_down/
    # 方策（行動を決めるネットワーク）
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,  # 観測の大きさのばらつきを自動でそろえる（推論用の .pt にも含まれる）
        # 試行錯誤の揺らぎの大きさ（最初の値）[rad]。0.1 で約 6° の揺らぎ
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    # 価値（今の状態がどれだけ良いかを見積もるネットワーク。学習中だけ使う）
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,  # 約 4 秒先の報酬まで考える（かがむ → 数秒後に完成ボーナス）
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
