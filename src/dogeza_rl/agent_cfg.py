"""PPO（RSL-RL）の学習方法の設定。Isaac Lab 付属タスクの設定を参考にしている。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DogezaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # 1 回の更新の前に、1 体あたり何ステップ経験を集めるか
    max_iterations = 3000  # 更新の回数
    save_interval = 100  # 何回ごとに途中経過（.pt）を保存するか
    experiment_name = "dogeza_g1"
    # 方策（行動を決めるネットワーク）
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,  # 観測の大きさのばらつきを自動でそろえる（推論用の .pt にも含まれる）
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
